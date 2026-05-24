#include "yolo_openvino.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <spdlog/spdlog.h>

#include "aligned_mat.hpp"
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
        rebuildAllowedClassTable();
        if (!initArmorClassifier(cfg, error))
        {
            return false;
        }
        auto model = core_.read_model(resolved_model_path);
        const auto model_input_shape = model->input().get_shape();
        if (model_input_shape.size() != 4 || model_input_shape[1] != 3)
        {
            if (error)
            {
                *error = "Unsupported model input shape, expected [N,3,H,W]";
            }
            return false;
        }
        input_h_ = static_cast<int>(model_input_shape[2]);
        input_w_ = static_cast<int>(model_input_shape[3]);

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
        postprocessor_.emplace(input_w_, input_h_, num_classes_, conf_thres_, nms_thres_);
        async_request_count_ = std::max(1, cfg.infer_workers);
        const int result_pool_size = cfg.result_message_pool_size > 0
                                         ? cfg.result_message_pool_size
                                         : async_request_count_ + 3;
        result_pool_.preallocate(static_cast<std::size_t>(result_pool_size));
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
            app::memory::CreateAlignedMat(slot.letterbox_img, input_h_, input_w_, CV_8UC3);
            slot.letterbox_img.setTo(cv::Scalar(114, 114, 114));
            slot.letterbox_padding_valid = true;
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
    slot.pipeline_start_tp = frame_msg.pipeline_start_tp;
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
    std::unique_lock<std::mutex> lock(completion_.mutex);
    if (!completion_.notified)
    {
        completion_.cv.wait_for(lock, timeout, [this]() {
            return completion_.notified || IsYoloStopRequested();
        });
    }
    completion_.notified = false;
}

void YoloOpenvino::notifyRequestCompleted(std::exception_ptr exception)
{
    {
        std::lock_guard<std::mutex> lock(completion_.mutex);
        if (exception && !completion_.exception)
        {
            completion_.exception = exception;
        }
        completion_.notified = true;
    }
    completion_.cv.notify_one();
}

bool YoloOpenvino::consumeCallbackException(std::string *error)
{
    std::exception_ptr exception;
    {
        std::lock_guard<std::mutex> lock(completion_.mutex);
        exception = completion_.exception;
        completion_.exception = nullptr;
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
        const double infer_async_ms = ElapsedMsSince(slot.submit_tp);
        app::profiling::Record(app::profiling::Stage::InferAsync, infer_async_ms);
        app::profiling::ScopedTimer post_timer(app::profiling::Stage::Postprocess);
        ov::Tensor out = slot.request.get_output_tensor();
        const AppConfig &cfg = GetAppConfig();
        std::vector<Detection> &dets = postprocessor_->run(slot.frame, out, slot.lb, cfg, class_allowed_);
        const bool enable_vis = IsFoxgloveDebugEnabled();
        const int candidate_limit = enable_vis ? std::min(static_cast<int>(dets.size()), max_color_candidates_)
                                               : static_cast<int>(dets.size());
        if (enable_vis)
        {
            const auto by_confidence_desc = [](const Detection &a, const Detection &b) {
                return a.confidence > b.confidence;
            };
            if (candidate_limit > 0 && candidate_limit < static_cast<int>(dets.size()))
            {
                std::partial_sort(dets.begin(),
                                  dets.begin() + candidate_limit,
                                  dets.end(),
                                  by_confidence_desc);
            }
            else
            {
                std::sort(dets.begin(), dets.end(), by_confidence_desc);
            }
        }
        cv::Mat vis;
        if (enable_vis)
        {
            app::memory::CreateAlignedMat(vis, slot.frame.rows, slot.frame.cols, slot.frame.type());
            slot.frame.copyTo(vis);
        }
        else
        {
            vis.release();
        }

        int kept_count = 0;
        bool has_best_corners = false;
        std::vector<ResultMParam::ArmorDetection> detected_armors;
        std::array<cv::Point2f, 4> best_corners = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f)};
        float best_conf = -1.0f;
        int best_class_id = -1;
        int best_armor_name_id = -1;
        float best_armor_name_confidence = 0.0f;
        std::string best_armor_name;
        const cv::Scalar box_color = (target_color_ == TargetColor::Red)
                                         ? cv::Scalar(0, 0, 255)
                                         : (target_color_ == TargetColor::Blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0));
        for (int i = 0; i < candidate_limit; ++i)
        {
            const auto &d = dets[static_cast<std::size_t>(i)];
            if (!d.has_keypoints)
            {
                continue;
            }
            ++kept_count;

            const std::array<cv::Point2f, 4> &corner_pts = d.keypoints;
            int armor_name_id = -1;
            float armor_name_confidence = 0.0f;
            std::string armor_name;
            classifyArmorNumber(slot.frame, corner_pts, armor_name_id, armor_name_confidence, armor_name);

            ResultMParam::ArmorDetection armor_det;
            armor_det.box = d.box;
            armor_det.class_id = d.class_id;
            armor_det.confidence = d.confidence;
            armor_det.armor_name_id = armor_name_id;
            armor_det.armor_name_confidence = armor_name_confidence;
            armor_det.armor_name = armor_name;
            armor_det.corners = corner_pts;
            detected_armors.push_back(std::move(armor_det));

            if (d.confidence > best_conf)
            {
                best_conf = d.confidence;
                best_class_id = d.class_id;
                best_armor_name_id = armor_name_id;
                best_armor_name_confidence = armor_name_confidence;
                best_armor_name = armor_name;
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
            int vis_name_id = -1;
            float vis_name_conf = 0.0f;
            std::string vis_name;
            if (classifyArmorNumber(slot.frame, corner_pts, vis_name_id, vis_name_conf, vis_name))
            {
                text += " num=" + vis_name + "(" + cv::format("%.2f", vis_name_conf) + ")";
            }
            int ty = std::max(0, d.box.y - 5);
            cv::putText(vis, text, cv::Point(d.box.x, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv::LINE_AA);
            DrawArmorFrame(vis, corner_pts, box_color);
        }
        result = result_pool_.acquire();
        result->vis = std::move(vis);
        result->frame_id = slot.frame_id;
        result->infer_id = slot.infer_id;
        result->detect_latency_ms = ElapsedMsSince(slot.capture_tp);
        result->latency_ms = ElapsedMsSince(slot.pipeline_start_tp);
        result->pipeline_start_tp = slot.pipeline_start_tp;
        result->capture_tp = slot.capture_tp;
        result->det_count = kept_count;
        result->class_id = best_class_id;
        result->armor_name_id = best_armor_name_id;
        result->armor_name_confidence = best_armor_name_confidence;
        result->armor_name = best_armor_name;
        result->has_corners = has_best_corners;
        result->corners = best_corners;
        result->armors = std::move(detected_armors);
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

bool YoloOpenvino::initArmorClassifier(const AppConfig &cfg, std::string *error)
{
    armor_classifier_enable_ = false;
    armor_classifier_confidence_ = cfg.armor_classifier_confidence;
    if (!cfg.armor_classifier_enable)
    {
        return true;
    }

    const std::string model_path = ResolveModelPath(cfg.armor_classifier_model_path);
    if (!std::filesystem::exists(model_path))
    {
        spdlog::warn("Armor number classifier disabled, model not found: {}", model_path);
        return true;
    }

    try
    {
        armor_classifier_ = cv::dnn::readNetFromONNX(model_path);
        armor_classifier_enable_ = !armor_classifier_.empty();
        spdlog::info("Armor number classifier loaded: {} threshold={:.2f}",
                     model_path,
                     armor_classifier_confidence_);
    }
    catch (const std::exception &e)
    {
        if (error)
        {
            *error = std::string("load armor number classifier failed: ") + e.what();
        }
        return false;
    }
    return true;
}

bool YoloOpenvino::classifyArmorNumber(const cv::Mat &frame,
                                       const std::array<cv::Point2f, 4> &corners,
                                       int &name_id,
                                       float &confidence,
                                       std::string &name) const
{
    name_id = -1;
    confidence = 0.0f;
    name.clear();
    if (!armor_classifier_enable_ || frame.empty())
    {
        return false;
    }

    cv::Rect rect = cv::boundingRect(std::vector<cv::Point2f>(corners.begin(), corners.end()));
    const int pad_x = std::max(2, rect.width / 8);
    const int pad_y = std::max(2, rect.height / 4);
    rect.x = std::max(0, rect.x - pad_x);
    rect.y = std::max(0, rect.y - pad_y);
    rect.width = std::min(frame.cols - rect.x, rect.width + 2 * pad_x);
    rect.height = std::min(frame.rows - rect.y, rect.height + 2 * pad_y);
    if (rect.width <= 2 || rect.height <= 2)
    {
        return false;
    }

    cv::Mat gray;
    cv::cvtColor(frame(rect), gray, cv::COLOR_BGR2GRAY);
    cv::Mat input(32, 32, CV_8UC1, cv::Scalar(0));
    const double scale = std::min(32.0 / static_cast<double>(gray.cols),
                                  32.0 / static_cast<double>(gray.rows));
    const int w = std::max(1, static_cast<int>(std::round(gray.cols * scale)));
    const int h = std::max(1, static_cast<int>(std::round(gray.rows * scale)));
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(w, h));
    const cv::Rect dst_roi(0, 0, std::min(32, w), std::min(32, h));
    resized(dst_roi).copyTo(input(dst_roi));

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());
    cv::dnn::Net net = armor_classifier_;
    net.setInput(blob);
    cv::Mat outputs = net.forward();
    if (outputs.empty())
    {
        return false;
    }
    outputs = outputs.reshape(1, 1);
    float max_logit = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max_logit, outputs);
    const float sum = static_cast<float>(cv::sum(outputs)[0]);
    if (sum <= 1e-6f)
    {
        return false;
    }
    outputs /= sum;

    double conf = 0.0;
    cv::Point label_point;
    cv::minMaxLoc(outputs, nullptr, &conf, nullptr, &label_point);
    const int id = label_point.x;
    if (id < 0 || id >= static_cast<int>(armor_name_labels_.size()))
    {
        return false;
    }
    name_id = id;
    confidence = static_cast<float>(conf);
    name = armor_name_labels_[static_cast<std::size_t>(id)];
    if (confidence < armor_classifier_confidence_ || name == "not_armor")
    {
        name_id = -1;
        name.clear();
        return false;
    }
    return true;
}

void YoloOpenvino::rebuildAllowedClassTable()
{
    class_allowed_.clear();
    if (target_color_ == TargetColor::Any)
    {
        return;
    }
    if (class_labels_.empty())
    {
        class_allowed_.resize(static_cast<std::size_t>(std::max(0, num_classes_)), 0U);
        return;
    }

    class_allowed_.resize(class_labels_.size(), 0U);
    for (std::size_t i = 0; i < class_labels_.size(); ++i)
    {
        const std::string &label = class_labels_[i];
        const bool is_red = (label == "r" || label == "red");
        const bool is_blue = (label == "b" || label == "blue");
        class_allowed_[i] = static_cast<uint8_t>(
            (target_color_ == TargetColor::Red && is_red) ||
            (target_color_ == TargetColor::Blue && is_blue));
    }
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
        slot.letterbox_padding_valid = false;
    }

    const int nw = slot.cached_resize_w;
    const int nh = slot.cached_resize_h;
    const int pad_w = slot.lb.pad_w;
    const int pad_h = slot.lb.pad_h;
    const bool needs_allocation = slot.letterbox_img.empty() ||
                                  slot.letterbox_img.rows != input_h_ ||
                                  slot.letterbox_img.cols != input_w_ ||
                                  slot.letterbox_img.type() != CV_8UC3 ||
                                  !app::memory::IsAligned(slot.letterbox_img.data);
    app::memory::CreateAlignedMat(slot.letterbox_img, input_h_, input_w_, CV_8UC3);
    if (needs_allocation)
    {
        slot.letterbox_padding_valid = false;
    }
    if (!slot.letterbox_padding_valid)
    {
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
        slot.letterbox_padding_valid = true;
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
