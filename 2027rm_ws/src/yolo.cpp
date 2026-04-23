#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include "camera_node.hpp"
#include "logging.hpp"
#include "yolo_app.hpp"

using namespace CGraph;

static const char *FRAME_TOPIC = "rm/frame/topic";
static const char *RESULT_TOPIC = "rm/result/topic";

#ifndef YOLO_CONFIG_PATH
#define YOLO_CONFIG_PATH "../config/yolo_config.yaml"
#endif

struct ColorBinaryConfig
{
    int roi_max_side = 96;
    int bright_threshold = 120;
    int dominance_threshold = 24;
    float min_color_ratio = 0.015f;
    float dominance_ratio = 1.1f;
    int morph_kernel = 3;
    int fast_roi_max_side = 48;
    int fast_bright_threshold = 90;
    int fast_dominance_threshold = 8;
    float fast_dominance_ratio = 1.08f;
};

enum class TargetColor
{
    Any,
    Red,
    Blue,
};

static AppConfig MakeDefaultAppConfig()
{
    return AppConfig{
        "model/last.xml",
        "CPU",
        0,
        0.25f,
        0.25f,
        4,
        2,
        "blue"};
}

static std::string ToLowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s;
}

static TargetColor ParseTargetColor(const std::string &color)
{
    const std::string target = ToLowerCopy(color);
    if (target == "red")
    {
        return TargetColor::Red;
    }
    if (target == "blue")
    {
        return TargetColor::Blue;
    }
    return TargetColor::Any;
}

static std::string ResolveTargetColorFromEnv()
{
    const char *v = std::getenv("RM_TARGET_COLOR");
    if (!v)
    {
        return "blue";
    }
    std::string color = ToLowerCopy(std::string(v));
    if (color == "red" || color == "blue")
    {
        return color;
    }
    return "blue";
}

static YAML::Node LoadYoloConfigWithFallback(std::string &used_path)
{
    const std::vector<std::string> candidates = {
        YOLO_CONFIG_PATH,
        "../config/yolo_config.yaml",
        "2027rm_ws/config/yolo_config.yaml",
        "config/yolo_config.yaml"};

    for (const auto &path : candidates)
    {
        std::ifstream fin(path);
        if (fin.good())
        {
            used_path = path;
            return YAML::LoadFile(path);
        }
    }
    throw std::runtime_error("bad file: yolo_config.yaml (tried absolute/relative fallback paths)");
}

static void LoadLoggingConfigNode(const YAML::Node &logging, app::logging::LogConfig &log_cfg)
{
    if (!logging)
    {
        return;
    }

    if (logging["level"]) log_cfg.level = ToLowerCopy(logging["level"].as<std::string>());
    if (logging["flush_level"]) log_cfg.flush_level = ToLowerCopy(logging["flush_level"].as<std::string>());
    if (logging["queue_size"]) log_cfg.queue_size = std::max<std::size_t>(1024, logging["queue_size"].as<std::size_t>());
    if (logging["thread_count"]) log_cfg.thread_count = std::max<std::size_t>(1, logging["thread_count"].as<std::size_t>());
    if (logging["overflow_policy"]) log_cfg.overflow_policy = ToLowerCopy(logging["overflow_policy"].as<std::string>());
    if (logging["pattern"]) log_cfg.pattern = logging["pattern"].as<std::string>();
}

static void LoadPerfLogSwitchNode(const YAML::Node &logging, bool &infer_fps_log_enable, bool &e2e_log_enable)
{
    if (!logging)
    {
        return;
    }

    if (logging["infer_fps_enable"]) infer_fps_log_enable = logging["infer_fps_enable"].as<bool>();
    if (logging["e2e_enable"]) e2e_log_enable = logging["e2e_enable"].as<bool>();
}

static app::logging::LogConfig LoadEarlyLoggingConfig()
{
    app::logging::LogConfig log_cfg;
    std::string used_path;
    try
    {
        YAML::Node root = LoadYoloConfigWithFallback(used_path);
        LoadLoggingConfigNode(root["logging"], log_cfg);
    }
    catch (...)
    {
    }
    return log_cfg;
}

static void LoadYoloConfig(AppConfig &cfg,
                           ColorBinaryConfig &color_cfg,
                           bool &foxglove_debug,
                           app::logging::LogConfig &log_cfg,
                           bool &infer_fps_log_enable,
                           bool &e2e_log_enable)
{
    std::string used_path;
    try
    {
        YAML::Node root = LoadYoloConfigWithFallback(used_path);
        LoadLoggingConfigNode(root["logging"], log_cfg);
        LoadPerfLogSwitchNode(root["logging"], infer_fps_log_enable, e2e_log_enable);
        app::logging::ApplyRuntimeLoggingConfig(log_cfg);

        const YAML::Node yolo = root["yolo"];
        if (yolo)
        {
            if (yolo["model_path"]) cfg.model_path = yolo["model_path"].as<std::string>();
            if (yolo["device"]) cfg.device = yolo["device"].as<std::string>();
            if (yolo["num_classes"]) cfg.num_classes = yolo["num_classes"].as<int>();
            if (yolo["conf_thres"]) cfg.conf_thres = yolo["conf_thres"].as<float>();
            if (yolo["nms_thres"]) cfg.nms_thres = yolo["nms_thres"].as<float>();
            if (yolo["thread_num"]) cfg.thread_num = std::max(2, yolo["thread_num"].as<int>());
            if (yolo["infer_workers"]) cfg.infer_workers = std::max(1, std::min(2, yolo["infer_workers"].as<int>()));
            if (yolo["target_color"]) cfg.target_color = ToLowerCopy(yolo["target_color"].as<std::string>());
        }

        const YAML::Node color = root["color_filter"];
        if (color)
        {
            if (color["roi_max_side"]) color_cfg.roi_max_side = std::max(16, color["roi_max_side"].as<int>());
            if (color["bright_threshold"]) color_cfg.bright_threshold = std::max(0, std::min(255, color["bright_threshold"].as<int>()));
            if (color["dominance_threshold"]) color_cfg.dominance_threshold = std::max(0, std::min(255, color["dominance_threshold"].as<int>()));
            if (color["min_color_ratio"]) color_cfg.min_color_ratio = std::max(0.0f, color["min_color_ratio"].as<float>());
            if (color["dominance_ratio"]) color_cfg.dominance_ratio = std::max(1.0f, color["dominance_ratio"].as<float>());
            if (color["morph_kernel"]) color_cfg.morph_kernel = std::max(1, color["morph_kernel"].as<int>());
            if (color["fast_roi_max_side"]) color_cfg.fast_roi_max_side = std::max(16, color["fast_roi_max_side"].as<int>());
            if (color["fast_bright_threshold"]) color_cfg.fast_bright_threshold = std::max(0, std::min(255, color["fast_bright_threshold"].as<int>()));
            if (color["fast_dominance_threshold"]) color_cfg.fast_dominance_threshold = std::max(0, std::min(255, color["fast_dominance_threshold"].as<int>()));
            if (color["fast_dominance_ratio"]) color_cfg.fast_dominance_ratio = std::max(1.0f, color["fast_dominance_ratio"].as<float>());
        }

        const YAML::Node debug = root["debug"];
        if (debug && debug["foxglove_enable"])
        {
            foxglove_debug = debug["foxglove_enable"].as<bool>();
        }

        if (cfg.target_color != "red" && cfg.target_color != "blue")
        {
            cfg.target_color = ResolveTargetColorFromEnv();
        }
        if ((color_cfg.morph_kernel % 2) == 0)
        {
            color_cfg.morph_kernel += 1;
        }
        spdlog::info("Loaded yolo config: {}", used_path);
    }
    catch (const std::exception &e)
    {
        cfg.target_color = ResolveTargetColorFromEnv();
        spdlog::warn("Load yolo config failed, using defaults: {}", e.what());
    }
}

static ColorBinaryConfig g_color_cfg{};

static bool PassFastColorGate(const cv::Mat &roi_bgr, TargetColor target)
{
    if (target == TargetColor::Any)
    {
        return true;
    }

    cv::Mat fast_roi = roi_bgr;
    const int max_side = g_color_cfg.fast_roi_max_side;
    if (fast_roi.cols > max_side || fast_roi.rows > max_side)
    {
        const float scale = std::min(static_cast<float>(max_side) / static_cast<float>(fast_roi.cols),
                                     static_cast<float>(max_side) / static_cast<float>(fast_roi.rows));
        const int new_w = std::max(3, static_cast<int>(std::round(fast_roi.cols * scale)));
        const int new_h = std::max(3, static_cast<int>(std::round(fast_roi.rows * scale)));
        cv::resize(fast_roi, fast_roi, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
    }

    const cv::Scalar mean_bgr = cv::mean(fast_roi);
    const float b = static_cast<float>(mean_bgr[0]);
    const float g = static_cast<float>(mean_bgr[1]);
    const float r = static_cast<float>(mean_bgr[2]);
    const float max_bg = std::max(b, g);
    const float max_rg = std::max(r, g);

    // Conservative fast gate: reject only when opposite color is clearly dominant.
    // If uncertain, keep candidate for the full per-pixel refinement to avoid false negatives.
    const bool red_strong = (r >= static_cast<float>(g_color_cfg.fast_bright_threshold)) &&
                            ((r - max_bg) >= static_cast<float>(g_color_cfg.fast_dominance_threshold)) &&
                            (r >= max_bg * g_color_cfg.fast_dominance_ratio);
    const bool blue_strong = (b >= static_cast<float>(g_color_cfg.fast_bright_threshold)) &&
                             ((b - max_rg) >= static_cast<float>(g_color_cfg.fast_dominance_threshold)) &&
                             (b >= max_rg * g_color_cfg.fast_dominance_ratio);

    if (target == TargetColor::Red)
    {
        if (blue_strong && !red_strong)
        {
            return false;
        }
        return true;
    }

    if (target == TargetColor::Blue)
    {
        if (red_strong && !blue_strong)
        {
            return false;
        }
        return true;
    }

    return true;
}

static bool IsBoxMatchedTargetColor(const cv::Mat &bgr, const cv::Rect &box, TargetColor target)
{
    if (bgr.empty())
    {
        return false;
    }

    const cv::Rect image_rect(0, 0, bgr.cols, bgr.rows);
    const cv::Rect roi = box & image_rect;
    if (roi.width < 3 || roi.height < 3)
    {
        return false;
    }

    cv::Mat roi_bgr = bgr(roi);
    if (!PassFastColorGate(roi_bgr, target))
    {
        return false;
    }

    const int max_side = g_color_cfg.roi_max_side;
    if (roi_bgr.cols > max_side || roi_bgr.rows > max_side)
    {
        const float scale = std::min(static_cast<float>(max_side) / static_cast<float>(roi_bgr.cols),
                                     static_cast<float>(max_side) / static_cast<float>(roi_bgr.rows));
        const int new_w = std::max(3, static_cast<int>(std::round(roi_bgr.cols * scale)));
        const int new_h = std::max(3, static_cast<int>(std::round(roi_bgr.rows * scale)));
        cv::resize(roi_bgr, roi_bgr, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
    }

    std::vector<cv::Mat> bgr_ch;
    cv::split(roi_bgr, bgr_ch);
    const cv::Mat &b = bgr_ch[0];
    const cv::Mat &g = bgr_ch[1];
    const cv::Mat &r = bgr_ch[2];

    cv::Mat max_bg, max_rg;
    cv::max(b, g, max_bg);
    cv::max(r, g, max_rg);

    cv::Mat red_dom, blue_dom;
    cv::subtract(r, max_bg, red_dom);
    cv::subtract(b, max_rg, blue_dom);

    cv::Mat bright_r, bright_b, red_mask, blue_mask;
    cv::threshold(r, bright_r, g_color_cfg.bright_threshold, 255, cv::THRESH_BINARY);
    cv::threshold(b, bright_b, g_color_cfg.bright_threshold, 255, cv::THRESH_BINARY);
    cv::threshold(red_dom, red_mask, g_color_cfg.dominance_threshold, 255, cv::THRESH_BINARY);
    cv::threshold(blue_dom, blue_mask, g_color_cfg.dominance_threshold, 255, cv::THRESH_BINARY);
    cv::bitwise_and(red_mask, bright_r, red_mask);
    cv::bitwise_and(blue_mask, bright_b, blue_mask);

    thread_local int cached_kernel_size = -1;
    thread_local cv::Mat kernel;
    if (cached_kernel_size != g_color_cfg.morph_kernel)
    {
        cached_kernel_size = g_color_cfg.morph_kernel;
        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(cached_kernel_size, cached_kernel_size));
    }
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, kernel);

    const int total_pixels = roi_bgr.cols * roi_bgr.rows;
    if (total_pixels <= 0)
    {
        return false;
    }

    const int red_pixels = cv::countNonZero(red_mask);
    const int blue_pixels = cv::countNonZero(blue_mask);
    const float red_ratio = static_cast<float>(red_pixels) / static_cast<float>(total_pixels);
    const float blue_ratio = static_cast<float>(blue_pixels) / static_cast<float>(total_pixels);

    if (red_ratio < g_color_cfg.min_color_ratio && blue_ratio < g_color_cfg.min_color_ratio)
    {
        return false;
    }

    if (target == TargetColor::Red)
    {
        return red_ratio > blue_ratio * g_color_cfg.dominance_ratio;
    }
    if (target == TargetColor::Blue)
    {
        return blue_ratio > red_ratio * g_color_cfg.dominance_ratio;
    }

    return true;
}

static bool g_foxglove_debug = false;
static app::logging::LogConfig g_log_cfg{};
static bool g_infer_fps_log_enable = true;
static bool g_e2e_log_enable = true;

static AppConfig g_cfg = []() {
    const app::logging::LogConfig early_log_cfg = LoadEarlyLoggingConfig();
    app::logging::InitAsyncLogging(early_log_cfg);

    AppConfig cfg = MakeDefaultAppConfig();
    LoadYoloConfig(cfg, g_color_cfg, g_foxglove_debug, g_log_cfg, g_infer_fps_log_enable, g_e2e_log_enable);
    return cfg;
}();

static std::atomic<bool> g_stop(false);

static bool IsFoxgloveDebugEnabled()
{
    return g_foxglove_debug;
}

static void DumpFrameForFoxglove(const cv::Mat& img)
{
    if (!IsFoxgloveDebugEnabled() || img.empty())
    {
        return;
    }

    static const std::string kDir = "/tmp/rm_rerun";
    static const std::string kTmp = kDir + "/latest.tmp.jpg";
    static const std::string kOut = kDir + "/latest.jpg";
    static bool init = false;
    if (!init)
    {
        std::error_code ec;
        std::filesystem::create_directories(kDir, ec);
        if (ec)
        {
            spdlog::warn("mkdir failed for {} error={}", kDir, ec.message());
        }
        init = true;
    }

    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
    if (cv::imwrite(kTmp, img, params))
    {
        std::rename(kTmp.c_str(), kOut.c_str());
    }
}

struct FrameMParam : public GMessageParam
{
    cv::Mat frame;
    uint64_t frame_id;
    int64_t ts_ms;
};

struct ResultMParam : public GMessageParam
{
    cv::Mat vis;
    uint64_t frame_id;
    int infer_id = 0;
    double latency_ms = 0.0;
    int det_count = 0;
    bool has_center = false;
    cv::Point2f center = cv::Point2f(0.0f, 0.0f);
};

struct Detection
{
    cv::Rect box;
    int class_id = -1;
    float confidence = 0.0f;
};

static cv::Point2f ComputeBoxCenter(const cv::Rect &box)
{
    return cv::Point2f(static_cast<float>(box.x) + static_cast<float>(box.width) * 0.5f,
                       static_cast<float>(box.y) + static_cast<float>(box.height) * 0.5f);
}

static int64_t NowMs()
{
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
};

static float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static float ToProb(float x)
{
    if (x >= 0.0f && x <= 1.0f)
    {
        return x;
    }
    return Sigmoid(x);
}

static std::string ResolveModelPath(const std::string &model_path)
{
    const std::vector<std::string> candidates = {
        model_path,
        std::string("2027rm_ws/") + model_path,
        std::string("../2027rm_ws/") + model_path,
        std::string("../") + model_path};

    for (const auto &path : candidates)
    {
        std::ifstream fin(path);
        if (fin.good())
        {
            return path;
        }
    }
    return model_path;
}

class YoloOpenvino
{
public:
    bool init(const std::string &model_path, const std::string &device, int num_classes, float conf_thres, float nms_thres, std::string *error)
    {
        try
        {
            num_classes_ = num_classes;
            conf_thres_ = conf_thres;
            nms_thres_ = nms_thres;
            const std::string resolved_model_path = ResolveModelPath(model_path);
            auto model = core_.read_model(resolved_model_path);
            // Explicit low-latency policy: prefer shortest response time over maximum throughput.
            compiled = core_.compile_model(
                model,
                device,
                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                ov::num_streams(1)
            );
            request_ = compiled.create_infer_request();
            input_port_ = compiled.input();
            output_port_ = compiled.output();
            spdlog::info("OpenVINO policy: device={} performance_mode=LATENCY num_streams=1", device);
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
            letterbox_img_.create(input_h_, input_w_, CV_8UC3);
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

    bool infer(const cv::Mat &frame,
               cv::Mat &vis,
               int &det_count,
               bool &has_center,
               cv::Point2f &center,
               std::string *error)
    {
        if (frame.empty())
        {
            if (error)
            {
                *error = "Input frame is empty";
            }
            return false;
        }
        preprocess(frame);
        ov::Tensor in_tensor(ov::element::f32,
                     ov::Shape{1, 3, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)},
                     blob_.ptr<float>());
        request_.set_input_tensor(in_tensor);
        request_.infer();
        ov::Tensor out = request_.get_output_tensor();
        std::vector<Detection> dets = postprocess(frame, out);
        const bool enable_vis = IsFoxgloveDebugEnabled();
        if (enable_vis)
        {
            vis = frame.clone();
        }
        else
        {
            vis.release();
        }

        int kept_count = 0;
        bool has_best_center = false;
        cv::Point2f best_center(0.0f, 0.0f);
        float best_conf = -1.0f;
        const TargetColor target = ParseTargetColor(g_cfg.target_color);
        const cv::Scalar box_color = (target == TargetColor::Red)
                                         ? cv::Scalar(0, 0, 255)
                                         : (target == TargetColor::Blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0));
        for (const auto &d : dets)
        {
            if (!IsBoxMatchedTargetColor(frame, d.box, target))
            {
                continue;
            }
            ++kept_count;

            const cv::Point2f center_pt = ComputeBoxCenter(d.box);
            if (d.confidence > best_conf)
            {
                best_conf = d.confidence;
                best_center = center_pt;
                has_best_center = true;
            }

            if (!enable_vis)
            {
                continue;
            }

            cv::rectangle(vis, d.box, box_color, 2);
            std::string text = "id=" + std::to_string(d.class_id) +
                               " conf=" + cv::format("%.2f", d.confidence);
            int ty = std::max(0, d.box.y - 5);
            cv::putText(vis, text, cv::Point(d.box.x, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv::LINE_AA);

            cv::circle(vis, center_pt, 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
        }
        det_count = kept_count;
        has_center = has_best_center;
        center = best_center;
        return true;
    }

private:
    struct LetterBoxInfo
    {
        float scale = 1.0f;
        int pad_w = 0;
        int pad_h = 0;
    };

    void preprocess(const cv::Mat &bgr)
    {
        const int src_w = bgr.cols;
        const int src_h = bgr.rows;
        lb_.scale = std::min(static_cast<float>(input_w_) / static_cast<float>(src_w),
                             static_cast<float>(input_h_) / static_cast<float>(src_h));
        const int nw = static_cast<int>(std::round(src_w * lb_.scale));
        const int nh = static_cast<int>(std::round(src_h * lb_.scale));
        lb_.pad_w = (input_w_ - nw) / 2;
        lb_.pad_h = (input_h_ - nh) / 2;

        letterbox_img_.setTo(cv::Scalar(114, 114, 114));
        cv::resize(bgr, resized_img_, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_LINEAR);
        resized_img_.copyTo(letterbox_img_(cv::Rect(lb_.pad_w, lb_.pad_h, nw, nh)));

        // Build NCHW float tensor with SIMD-optimized OpenCV path (scale + swapRB + layout transform).
        cv::dnn::blobFromImage(letterbox_img_, blob_, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    }
    std::vector<Detection> postprocess(const cv::Mat &orig, ov::Tensor &out) const
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> scores;
        auto shape = out.get_shape();
        const float *data = out.data<float>();
        if (shape.size() != 3)
            return {};

        const bool channel_first = (shape[1] < shape[2]);
        const int C = static_cast<int>(channel_first ? shape[1] : shape[2]);
        const int N = static_cast<int>(channel_first ? shape[2] : shape[1]);
        boxes.reserve(static_cast<size_t>(N));
        class_ids.reserve(static_cast<size_t>(N));
        scores.reserve(static_cast<size_t>(N));

        auto at = [&](int c, int i) -> float
        {
            if (channel_first)
            {
                return data[c * N + i];
            }
            return data[i * C + c];
        };

        auto decode_box = [&](float cx, float cy, float w, float h, int cls, float score)
        {
            float x1 = cx - 0.5f * w;
            float y1 = cy - 0.5f * h;
            float x2 = cx + 0.5f * w;
            float y2 = cy + 0.5f * h;
            x1 = (x1 - lb_.pad_w) / lb_.scale;
            y1 = (y1 - lb_.pad_h) / lb_.scale;
            x2 = (x2 - lb_.pad_w) / lb_.scale;
            y2 = (y2 - lb_.pad_h) / lb_.scale;
            int left = std::max(0, std::min(static_cast<int>(std::round(x1)), orig.cols - 1));
            int top = std::max(0, std::min(static_cast<int>(std::round(y1)), orig.rows - 1));
            int right = std::max(0, std::min(static_cast<int>(std::round(x2)), orig.cols - 1));
            int bottom = std::max(0, std::min(static_cast<int>(std::round(y2)), orig.rows - 1));
            if (right <= left || bottom <= top)
                return;
            boxes.emplace_back(left, top, right - left, bottom - top);
            class_ids.push_back(cls);
            scores.push_back(score);
        };

        // This model outputs decoded boxes as [cx, cy, w, h], then class probabilities,
        // then extra attributes (e.g. keypoints). We only use class channels.
        const std::vector<int> strides = {8, 16, 32};
        int expected_points = 0;
        for (int s : strides)
        {
            expected_points += (input_h_ / s) * (input_w_ / s);
        }

        if (C >= 6 && N == expected_points)
        {
            int inferred_cls = 0;
            const int sample_n = std::min(N, 256);
            for (int c = 4; c < C; ++c)
            {
                int in01 = 0;
                for (int i = 0; i < sample_n; ++i)
                {
                    float v = at(c, i);
                    if (v >= 0.0f && v <= 1.0f)
                        ++in01;
                }
                float ratio = static_cast<float>(in01) / static_cast<float>(sample_n);
                if (ratio >= 0.98f)
                    ++inferred_cls;
                else
                    break;
            }

            int cls_count = std::max(0, C - 4);
            if (inferred_cls > 0)
                cls_count = inferred_cls;
            int use_cls = (num_classes_ > 0) ? std::min(num_classes_, cls_count) : cls_count;

            for (int i = 0; i < N; ++i)
            {
                float cx = at(0, i);
                float cy = at(1, i);
                float w = at(2, i);
                float h = at(3, i);

                int best_cls = -1;
                float best_score = 0.0f;
                for (int c = 0; c < use_cls; ++c)
                {
                    float s = ToProb(at(4 + c, i));
                    if (s > best_score)
                    {
                        best_score = s;
                        best_cls = c;
                    }
                }

                if (best_score < conf_thres_)
                    continue;
                decode_box(cx, cy, w, h, best_cls, best_score);
            }
        }
        else
        {
            int cls_count = std::max(0, C - 4);
            int use_cls = (num_classes_ > 0) ? std::min(num_classes_, cls_count) : cls_count;
            for (int i = 0; i < N; ++i)
            {
                float cx = at(0, i);
                float cy = at(1, i);
                float w = at(2, i);
                float h = at(3, i);
                int best_cls = -1;
                float best_score = 0.0f;
                for (int c = 0; c < use_cls; ++c)
                {
                    float s = ToProb(at(4 + c, i));
                    if (s > best_score)
                    {
                        best_score = s;
                        best_cls = c;
                    }
                }
                if (best_score < conf_thres_)
                    continue;
                decode_box(cx, cy, w, h, best_cls, best_score);
            }
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres_, nms_thres_, keep);
        std::vector<Detection> dets;
        dets.reserve(keep.size());
        for (int idx : keep)
        {
            Detection d;
            d.box = boxes[idx];
            d.class_id = class_ids[idx];
            d.confidence = scores[idx];
            dets.push_back(d);
        }
        return dets;
    }

private:
    ov::Core core_;
    ov::CompiledModel compiled;
    ov::InferRequest request_;
    ov::Output<const ov::Node> input_port_;
    ov::Output<const ov::Node> output_port_;
    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 80;
    float conf_thres_ = 0.25f;
    float nms_thres_ = 0.45f;
    mutable LetterBoxInfo lb_;
    cv::Mat letterbox_img_;
    cv::Mat resized_img_;
    cv::Mat blob_;
};
class CameraPubNode : public GNode
{
private:
    HikCameraNode camera_;
    uint64_t frame_id_ = 0;

public:
    CStatus init() override
    {
        std::string err;
        if (!camera_.init(&err))
            return CStatus(err);
        frame_id_ = 0;
        return CStatus();
    }
    CStatus run() override
    {
        while (!g_stop.load())
        {
            cv::Mat frame;
            std::string err;
            if (!camera_.grab(frame, &err))
            {
                return CStatus(err);
            }
            std::shared_ptr<FrameMParam> msg(new FrameMParam());
            msg->frame = std::move(frame);
            msg->frame_id = ++frame_id_;
            msg->ts_ms = NowMs();
            CStatus st = CGRAPH_SEND_MPARAM(FrameMParam, FRAME_TOPIC, msg, GMessagePushStrategy::REPLACE);
            if (st.isErr())
                return st;
        }
        return CStatus();
    }
    CStatus destroy() override
    {
        camera_.shutdown();
        return CStatus();
    }
};
class YoloInferNode : public GNode
{
private:
    YoloOpenvino yolo_;

private:
    static int parseInferId(const std::string &name)
    {
        if (name.find("infer_1") != std::string::npos)
            return 1;
        if (name.find("infer_2") != std::string::npos)
            return 2;
        return -1;
    }

public:
    CStatus init() override
    {
        std::string err;
        if (!yolo_.init(g_cfg.model_path, g_cfg.device, g_cfg.num_classes, g_cfg.conf_thres, g_cfg.nms_thres, &err))
        {
            return CStatus(err);
        }
        return CStatus();
    }
    CStatus run() override
    {
        const int infer_id = parseInferId(this->getName());
        auto infer_log_start = std::chrono::steady_clock::now();
        int infer_count = 0;
        double infer_ms_sum = 0.0;
        while (!g_stop.load())
        {
            std::shared_ptr<FrameMParam> in = nullptr;
            CStatus st = CGRAPH_RECV_MPARAM_WITH_TIMEOUT(FrameMParam, FRAME_TOPIC, in, 1000);
            if (st.isErr())
            {
                if (g_stop.load())
                    break;
                continue;
            }
            if (!in || in->frame.empty())
                continue;
            cv::Mat vis;
            int det_count = 0;
            bool has_center = false;
            cv::Point2f center(0.0f, 0.0f);
            std::string err;

            const auto infer_begin = std::chrono::steady_clock::now();
            if (!yolo_.infer(in->frame, vis, det_count, has_center, center, &err))
            {
                return CStatus(err);
            }
            const auto infer_end = std::chrono::steady_clock::now();
            const double infer_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_begin).count());
            ++infer_count;
            infer_ms_sum += infer_ms;

            const auto infer_log_dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_log_start).count();
            if (g_infer_fps_log_enable && infer_log_dt_ms >= 1000)
            {
                const double infer_fps = infer_count * 1000.0 / static_cast<double>(infer_log_dt_ms);
                const double infer_ms_avg = (infer_count > 0) ? (infer_ms_sum / static_cast<double>(infer_count)) : 0.0;
                spdlog::info("[YOLO_INFER_FPS] infer={} fps={} infer_ms_avg={}",
                             infer_id,
                             static_cast<int>(infer_fps + 0.5),
                             static_cast<int>(infer_ms_avg + 0.5));
                infer_count = 0;
                infer_ms_sum = 0.0;
                infer_log_start = infer_end;
            }

            std::shared_ptr<ResultMParam> out(new ResultMParam());
            out->vis = vis;
            out->frame_id = in->frame_id;
            out->infer_id = infer_id;
            out->latency_ms = static_cast<double>(NowMs() - in->ts_ms);
            out->det_count = det_count;
            out->has_center = has_center;
            out->center = center;
            st = CGRAPH_SEND_MPARAM(ResultMParam, RESULT_TOPIC, out, GMessagePushStrategy::REPLACE);
            if (st.isErr())
                return st;
        }
        return CStatus();
    }
};

class DisplayNode : public GNode {
public:
    CStatus init() override {
        fps_start_ = std::chrono::steady_clock::now();
        log_start_ = fps_start_;
        count_ = 0;
        log_count_ = 0;
        fps_ = 0.0;
        latency_sum_ms_ = 0.0;
        latency_max_ms_ = 0.0;
        return CStatus();
    }

    CStatus run() override {
        const bool enable_vis = IsFoxgloveDebugEnabled();
        while (!g_stop.load()) {
            std::shared_ptr<ResultMParam> r = nullptr;
            CStatus st = CGRAPH_RECV_MPARAM_WITH_TIMEOUT(ResultMParam, RESULT_TOPIC, r, 1000);
            if (st.isErr()) {
                if (g_stop.load()) break;
                continue;
            }
            if (!r) continue;
            ++count_;
            ++log_count_;
            latency_sum_ms_ += r->latency_ms;
            latency_max_ms_ = std::max(latency_max_ms_, r->latency_ms);
            auto now = std::chrono::steady_clock::now();
            auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start_).count();
            if (dt_ms >= 1000) {
                fps_ = count_ * 1000.0 / static_cast<double>(dt_ms);
                count_ = 0;
                fps_start_ = now;
            }

            auto log_dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - log_start_).count();
            if (g_e2e_log_enable && log_dt_ms >= 1000) {
                double real_fps = log_count_ * 1000.0 / static_cast<double>(log_dt_ms);
                const double latency_avg_ms = (log_count_ > 0) ? (latency_sum_ms_ / static_cast<double>(log_count_)) : 0.0;
                spdlog::info("[E2E] fps={} latency_ms_avg={} latency_ms_max={} det={} infer={}",
                             static_cast<int>(real_fps + 0.5),
                             static_cast<int>(latency_avg_ms + 0.5),
                             static_cast<int>(latency_max_ms_ + 0.5),
                             r->det_count,
                             r->infer_id);
                log_count_ = 0;
                latency_sum_ms_ = 0.0;
                latency_max_ms_ = 0.0;
                log_start_ = now;
            }

            if (!enable_vis || r->vis.empty()) {
                continue;
            }

            cv::Mat &show = r->vis;
            std::string t1 = "frame=" + std::to_string(r->frame_id) + " infer=" + std::to_string(r->infer_id) + " det=" + std::to_string(r->det_count);
            std::string t2 = "fps=" + std::to_string(static_cast<int>(fps_ + 0.5));
            std::string t3 = "latency=" + std::to_string(static_cast<int>(r->latency_ms + 0.5)) + "ms";
            std::string t4 = "center=none";
            if (r->has_center)
            {
                t4 = "center=(" + std::to_string(static_cast<int>(std::round(r->center.x))) +
                     "," + std::to_string(static_cast<int>(std::round(r->center.y))) + ")";
                cv::circle(show, r->center, 6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
            cv::putText(show, t1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            cv::putText(show, t2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(show, t3, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
            cv::putText(show, t4, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            DumpFrameForFoxglove(show);
        }
        return CStatus();
    }

    CStatus destroy() override {
        return CStatus();
    }

private:
    std::chrono::steady_clock::time_point fps_start_;
    std::chrono::steady_clock::time_point log_start_;
    int count_ = 0;
    int log_count_ = 0;
    double fps_ = 0.0;
    double latency_sum_ms_ = 0.0;
    double latency_max_ms_ = 0.0;
};

const AppConfig& GetAppConfig() {
    return g_cfg;
}

CStatus RegisterYoloPipelineElements(CGraph::GPipeline* const &pipeline) {
    GElementPtr cam = nullptr;
    GElementPtr infer1 = nullptr;
    GElementPtr infer2 = nullptr;
    GElementPtr display = nullptr;
    CStatus st;
    st += pipeline->registerGElement<CameraPubNode>(&cam, {}, "camera_pub");
    st += pipeline->registerGElement<YoloInferNode>(&infer1, {}, "infer_1");
    if (g_cfg.infer_workers >= 2) {
        st += pipeline->registerGElement<YoloInferNode>(&infer2, {}, "infer_2");
    }
    st += pipeline->registerGElement<DisplayNode>(&display, {}, "display");
    return st;
}

void InitYoloMessageTopics() {
    CGRAPH_CREATE_MESSAGE_TOPIC(FrameMParam, FRAME_TOPIC, 2);
    CGRAPH_CREATE_MESSAGE_TOPIC(ResultMParam, RESULT_TOPIC, 2);
}

void ShutdownYoloApp() {
    g_stop.store(true);
    CGRAPH_CLEAR_MESSAGES()
}