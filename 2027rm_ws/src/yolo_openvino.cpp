#include "yolo_openvino.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

#include "yolo_common.hpp"

bool YoloOpenvino::init(const AppConfig &cfg, std::string *error)
{
    try
    {
        num_classes_ = cfg.num_classes;
        conf_thres_ = cfg.conf_thres;
        nms_thres_ = cfg.nms_thres;
        const std::string resolved_model_path = ResolveModelPath(cfg.model_path);
        class_labels_ = LoadModelLabels(resolved_model_path);
        auto model = core_.read_model(resolved_model_path);
        compiled = core_.compile_model(
            model,
            cfg.device,
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::num_streams(1));
        input_port_ = compiled.input();
        output_port_ = compiled.output();
        if (!class_labels_.empty())
        {
            std::string labels_desc;
            for (std::size_t i = 0; i < class_labels_.size(); ++i)
            {
                if (i > 0)
                {
                    labels_desc += ",";
                }
                labels_desc += std::to_string(i) + ":" + class_labels_[i];
            }
            spdlog::info("Model labels: {}", labels_desc);
        }
        auto in_shape = input_port_.get_shape();
        if (in_shape.size() != 4 || in_shape[1] != 3)
        {
            if (error)
            {
                *error = "Unsupported input shape, expected [N,3,H,W]";
            }
            return false;
        }
        input_h_ = static_cast<int>(in_shape[2]);
        input_w_ = static_cast<int>(in_shape[3]);
        postprocessor_.emplace(input_w_, input_h_, num_classes_, conf_thres_, nms_thres_);
        async_request_count_ = std::max(1, cfg.infer_workers);
        slots_.clear();
        slots_.reserve(static_cast<std::size_t>(async_request_count_));
        for (int i = 0; i < async_request_count_; ++i)
        {
            AsyncSlot slot;
            slot.request = compiled.create_infer_request();
            slot.infer_id = i + 1;
            slot.blob.create(1, 3 * input_h_ * input_w_, CV_32F);
            slots_.push_back(std::move(slot));
        }
        spdlog::info("OpenVINO policy: device={} performance_mode=LATENCY num_streams=1 async_requests={}",
                     cfg.device,
                     async_request_count_);
        return true;
    }
    catch (const std::exception &e)
    {
        if (error)
        {
            *error = e.what();
        }
        return false;
    }
}

bool YoloOpenvino::hasIdleRequest() const
{
    return std::any_of(slots_.begin(), slots_.end(), [](const AsyncSlot &slot) {
        return !slot.busy;
    });
}

bool YoloOpenvino::submit(const FrameMParam &frame_msg, std::string *error)
{
    if (frame_msg.frame.empty())
    {
        if (error)
        {
            *error = "Input frame is empty";
        }
        return false;
    }

    auto slot_it = std::find_if(slots_.begin(), slots_.end(), [](const AsyncSlot &slot) {
        return !slot.busy;
    });
    if (slot_it == slots_.end())
    {
        return true;
    }

    AsyncSlot &slot = *slot_it;
    slot.frame = frame_msg.frame;
    slot.frame_id = frame_msg.frame_id;
    slot.capture_tp = frame_msg.capture_tp;
    preprocess(slot.frame, slot.resized_img, slot.blob, slot.lb);
    ov::Tensor in_tensor(ov::element::f32,
                         ov::Shape{1, 3, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)},
                         slot.blob.ptr<float>());
    slot.request.set_input_tensor(in_tensor);
    slot.request.start_async();
    slot.busy = true;
    return true;
}

bool YoloOpenvino::collectCompleted(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error)
{
    for (AsyncSlot &slot : slots_)
    {
        if (!slot.busy)
        {
            continue;
        }

        bool ready = false;
        try
        {
            ready = slot.request.wait_for(std::chrono::milliseconds(0));
        }
        catch (const std::exception &e)
        {
            if (error)
            {
                *error = e.what();
            }
            return false;
        }

        if (!ready)
        {
            continue;
        }

        std::shared_ptr<ResultMParam> result;
        if (!finishSlot(slot, result, error))
        {
            return false;
        }
        results.push_back(result);
    }
    return true;
}

bool YoloOpenvino::waitAll(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error)
{
    for (AsyncSlot &slot : slots_)
    {
        if (!slot.busy)
        {
            continue;
        }
        try
        {
            slot.request.wait();
        }
        catch (const std::exception &e)
        {
            if (error)
            {
                *error = e.what();
            }
            return false;
        }
        std::shared_ptr<ResultMParam> result;
        if (!finishSlot(slot, result, error))
        {
            return false;
        }
        results.push_back(result);
    }
    return true;
}

bool YoloOpenvino::finishSlot(AsyncSlot &slot, std::shared_ptr<ResultMParam> &result, std::string *error)
{
    try
    {
        ov::Tensor out = slot.request.get_output_tensor();
        const TargetColor target = ParseTargetColor(GetAppConfig().target_color);
        std::vector<Detection> dets = postprocessor_->run(slot.frame, out, slot.lb, GetAppConfig(), [&](int class_id) {
            return MatchTargetColorByClass(class_id, target);
        });
        std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) {
            return a.confidence > b.confidence;
        });
        const bool enable_vis = IsFoxgloveDebugEnabled();
        cv::Mat vis;
        if (enable_vis)
        {
            vis = slot.frame.clone();
        }
        else
        {
            vis.release();
        }

        int kept_count = 0;
        bool has_best_corners = false;
        std::array<cv::Point2f, 4> best_corners = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f)};
        float best_conf = -1.0f;
        const cv::Scalar box_color = (target == TargetColor::Red)
                                         ? cv::Scalar(0, 0, 255)
                                         : (target == TargetColor::Blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0));
        const int candidate_limit = std::min(static_cast<int>(dets.size()), std::max(1, GetAppConfig().max_color_candidates));
        for (int i = 0; i < candidate_limit; ++i)
        {
            const auto &d = dets[static_cast<std::size_t>(i)];
            if (!d.has_keypoints)
            {
                continue;
            }
            ++kept_count;

            const std::array<cv::Point2f, 4> corner_pts = OrderCorners(d.keypoints);
            if (d.confidence > best_conf)
            {
                best_conf = d.confidence;
                best_corners = corner_pts;
                has_best_corners = true;
                if (!enable_vis)
                {
                    break;
                }
            }

            if (!enable_vis)
            {
                continue;
            }

            std::string text = "id=" + std::to_string(d.class_id) +
                               " conf=" + cv::format("%.2f", d.confidence) +
                               " kpt";
            int ty = std::max(0, d.box.y - 5);
            cv::putText(vis, text, cv::Point(d.box.x, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv::LINE_AA);
            DrawArmorFrame(vis, corner_pts, box_color);
        }
        result.reset(new ResultMParam());
        result->vis = std::move(vis);
        result->frame_id = slot.frame_id;
        result->infer_id = slot.infer_id;
        result->latency_ms = ElapsedMsSince(slot.capture_tp);
        result->capture_tp = slot.capture_tp;
        result->det_count = kept_count;
        result->has_corners = has_best_corners;
        result->corners = best_corners;
        slot.busy = false;
        slot.frame.release();
    }
    catch (const std::exception &e)
    {
        slot.busy = false;
        if (error)
        {
            *error = e.what();
        }
        return false;
    }
    return true;
}

bool YoloOpenvino::MatchTargetColorByClass(int class_id, TargetColor target) const
{
    if (target == TargetColor::Any)
    {
        return true;
    }
    if (class_id < 0 || static_cast<std::size_t>(class_id) >= class_labels_.size())
    {
        return false;
    }

    const std::string &label = class_labels_[static_cast<std::size_t>(class_id)];
    const bool is_red = (label == "r" || label == "red");
    const bool is_blue = (label == "b" || label == "blue");
    if (target == TargetColor::Red)
    {
        return is_red;
    }
    if (target == TargetColor::Blue)
    {
        return is_blue;
    }
    return true;
}

void YoloOpenvino::preprocess(const cv::Mat &bgr, cv::Mat &resized_img, cv::Mat &blob, LetterBoxInfo &lb) const
{
    const int src_w = bgr.cols;
    const int src_h = bgr.rows;
    lb.scale = std::min(static_cast<float>(input_w_) / static_cast<float>(src_w),
                        static_cast<float>(input_h_) / static_cast<float>(src_h));
    const int nw = static_cast<int>(std::round(src_w * lb.scale));
    const int nh = static_cast<int>(std::round(src_h * lb.scale));
    lb.pad_w = (input_w_ - nw) / 2;
    lb.pad_h = (input_h_ - nh) / 2;

    cv::resize(bgr, resized_img, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_LINEAR);

    const int plane_size = input_h_ * input_w_;
    blob.create(1, 3 * plane_size, CV_32F);
    float *blob_ptr = blob.ptr<float>();
    std::fill(blob_ptr, blob_ptr + (3 * plane_size), 114.0f / 255.0f);

    float *dst_r = blob_ptr;
    float *dst_g = blob_ptr + plane_size;
    float *dst_b = blob_ptr + 2 * plane_size;
    for (int y = 0; y < nh; ++y)
    {
        const cv::Vec3b *src_row = resized_img.ptr<cv::Vec3b>(y);
        const int dst_y = y + lb.pad_h;
        const int dst_offset = dst_y * input_w_ + lb.pad_w;
        for (int x = 0; x < nw; ++x)
        {
            const cv::Vec3b &pixel = src_row[x];
            const int idx = dst_offset + x;
            dst_r[idx] = static_cast<float>(pixel[2]) / 255.0f;
            dst_g[idx] = static_cast<float>(pixel[1]) / 255.0f;
            dst_b[idx] = static_cast<float>(pixel[0]) / 255.0f;
        }
    }
}

void YoloResultFilter::EmaScalar::reset()
{
    initialized_ = false;
    prev_ = 0.0;
}

double YoloResultFilter::EmaScalar::filter(double x, double alpha)
{
    if (!initialized_)
    {
        initialized_ = true;
        prev_ = x;
        return x;
    }

    const double filtered = alpha * x + (1.0 - alpha) * prev_;
    prev_ = filtered;
    return filtered;
}

void YoloResultFilter::reset()
{
    initialized_ = false;
    last_frame_id_ = 0;
    for (auto &filter : filters_)
    {
        filter.reset();
    }
}

void YoloResultFilter::apply(ResultMParam &result, const AppConfig &cfg)
{
    if (!result.has_corners)
    {
        reset();
        return;
    }
    if (!cfg.ema_enable)
    {
        return;
    }

    if (!initialized_ ||
        result.frame_id <= last_frame_id_ ||
        result.frame_id - last_frame_id_ > cfg.ema_reset_frame_gap)
    {
        reset();
        initialized_ = true;
    }

    const double alpha = std::clamp(cfg.ema_alpha, 0.0, 1.0);
    for (std::size_t i = 0; i < result.corners.size(); ++i)
    {
        result.corners[i].x = static_cast<float>(filters_[i * 2].filter(result.corners[i].x, alpha));
        result.corners[i].y = static_cast<float>(filters_[i * 2 + 1].filter(result.corners[i].y, alpha));
    }

    last_frame_id_ = result.frame_id;
}
