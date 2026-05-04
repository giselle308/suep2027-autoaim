#include "yolo_openvino.hpp"

#include <algorithm>
#include <cmath>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <spdlog/spdlog.h>

#include "yolo_common.hpp"
#include "profiling.hpp"

bool YoloOpenvino::init(const AppConfig &cfg, std::string *error)
{
    try
    {
        num_classes_ = cfg.num_classes;
        conf_thres_ = cfg.conf_thres;
        nms_thres_ = cfg.nms_thres;
        target_color_ = ParseTargetColor(cfg.target_color);
        max_color_candidates_ = std::max(1, cfg.max_color_candidates);
        const std::string resolved_model_path = ResolveModelPath(cfg.model_path);
        class_labels_ = LoadModelLabels(resolved_model_path);
        auto model = core_.read_model(resolved_model_path);
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().model().set_layout("NCHW");
        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0f);
        model = ppp.build();
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
            slot.request.set_callback([this](std::exception_ptr exception) {
                notifyRequestCompleted(exception);
            });
            slot.infer_id = i + 1;
            slots_.push_back(std::move(slot));
        }
        spdlog::info("OpenVINO policy: device={} performance_mode=LATENCY num_streams=1 async_requests={} preprocess=u8_nhwc_bgr",
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
    {
        app::profiling::ScopedTimer preprocess_timer(app::profiling::Stage::Preprocess);
        preprocess(slot);
    }
    ov::Tensor in_tensor(ov::element::u8,
                         ov::Shape{1, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_), 3},
                         slot.letterbox_img.data);
    slot.request.set_input_tensor(in_tensor);
    slot.submit_tp = std::chrono::steady_clock::now();
    slot.request.start_async();
    slot.busy = true;
    return true;
}

bool YoloOpenvino::collectCompleted(std::vector<std::shared_ptr<ResultMParam>> &results, std::string *error)
{
    if (!consumeCallbackException(error))
    {
        return false;
    }

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

void YoloOpenvino::waitForCompletionSignal(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(completion_mutex_);
    if (!completion_notified_)
    {
        completion_cv_.wait_for(lock, timeout, [this]() {
            return completion_notified_ || IsYoloStopRequested();
        });
    }
    completion_notified_ = false;
}

void YoloOpenvino::notifyRequestCompleted(std::exception_ptr exception)
{
    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (exception && !callback_exception_)
        {
            callback_exception_ = exception;
        }
        completion_notified_ = true;
    }
    completion_cv_.notify_one();
}

bool YoloOpenvino::consumeCallbackException(std::string *error)
{
    std::exception_ptr exception;
    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        exception = callback_exception_;
        callback_exception_ = nullptr;
    }
    if (!exception)
    {
        return true;
    }

    try
    {
        std::rethrow_exception(exception);
    }
    catch (const std::exception &e)
    {
        if (error)
        {
            *error = e.what();
        }
    }
    catch (...)
    {
        if (error)
        {
            *error = "OpenVINO async inference failed";
        }
    }
    return false;
}

bool YoloOpenvino::finishSlot(AsyncSlot &slot, std::shared_ptr<ResultMParam> &result, std::string *error)
{
    try
    {
        app::profiling::Record(app::profiling::Stage::InferAsync, ElapsedMsSince(slot.submit_tp));
        app::profiling::ScopedTimer post_timer(app::profiling::Stage::Postprocess);
        ov::Tensor out = slot.request.get_output_tensor();
        const AppConfig &cfg = GetAppConfig();
        std::vector<Detection> dets = postprocessor_->run(slot.frame, out, slot.lb, cfg, [&](int class_id) {
            return MatchTargetColorByClass(class_id, target_color_);
        });
        const bool enable_vis = IsFoxgloveDebugEnabled();
        if (enable_vis)
        {
            std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) {
                return a.confidence > b.confidence;
            });
        }
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
        const cv::Scalar box_color = (target_color_ == TargetColor::Red)
                                         ? cv::Scalar(0, 0, 255)
                                         : (target_color_ == TargetColor::Blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0));
        const int candidate_limit = enable_vis ? std::min(static_cast<int>(dets.size()), max_color_candidates_)
                                               : static_cast<int>(dets.size());
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
        result = result_pool_.acquire();
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

void YoloOpenvino::preprocess(AsyncSlot &slot) const
{
    const cv::Mat &bgr = slot.frame;
    const int src_w = bgr.cols;
    const int src_h = bgr.rows;
    const cv::Size src_size(src_w, src_h);
    if (!slot.letterbox_cache_valid || slot.cached_src_size != src_size)
    {
        slot.lb.scale = std::min(static_cast<float>(input_w_) / static_cast<float>(src_w),
                                 static_cast<float>(input_h_) / static_cast<float>(src_h));
        slot.cached_resize_w = static_cast<int>(std::round(src_w * slot.lb.scale));
        slot.cached_resize_h = static_cast<int>(std::round(src_h * slot.lb.scale));
        slot.lb.pad_w = (input_w_ - slot.cached_resize_w) / 2;
        slot.lb.pad_h = (input_h_ - slot.cached_resize_h) / 2;
        slot.cached_src_size = src_size;
        slot.letterbox_cache_valid = true;
    }

    const int nw = slot.cached_resize_w;
    const int nh = slot.cached_resize_h;
    const int pad_w = slot.lb.pad_w;
    const int pad_h = slot.lb.pad_h;
    slot.letterbox_img.create(input_h_, input_w_, CV_8UC3);
    if (pad_h > 0)
    {
        slot.letterbox_img(cv::Rect(0, 0, input_w_, pad_h)).setTo(cv::Scalar(114, 114, 114));
        slot.letterbox_img(cv::Rect(0, pad_h + nh, input_w_, input_h_ - pad_h - nh)).setTo(cv::Scalar(114, 114, 114));
    }
    if (pad_w > 0)
    {
        slot.letterbox_img(cv::Rect(0, pad_h, pad_w, nh)).setTo(cv::Scalar(114, 114, 114));
        slot.letterbox_img(cv::Rect(pad_w + nw, pad_h, input_w_ - pad_w - nw, nh)).setTo(cv::Scalar(114, 114, 114));
    }

    cv::Mat roi = slot.letterbox_img(cv::Rect(pad_w, pad_h, nw, nh));
    cv::resize(bgr, roi, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_LINEAR);
    slot.resized_img = roi;

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
