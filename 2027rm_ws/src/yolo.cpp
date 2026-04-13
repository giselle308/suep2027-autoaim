#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <array>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <yaml-cpp/yaml.h>
#include "camera_node.hpp"
#include "yolo_app.hpp"

using namespace CGraph;

static const char *FRAME_TOPIC = "rm/frame/topic";
static const char *RESULT_TOPIC = "rm/result/topic";

#ifndef YOLO_CONFIG_PATH
#define YOLO_CONFIG_PATH "../config/yolo_config.yaml"
#endif

enum class LightColor
{
    UNKNOWN = 0,
    RED = 1,
    BLUE = 2,
};

struct ColorBinaryConfig
{
    int roi_max_side = 96;
    int bright_threshold = 120;
    int dominance_threshold = 24;
    float min_color_ratio = 0.015f;
    float dominance_ratio = 1.1f;
    int morph_kernel = 3;
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

static void LoadYoloConfig(AppConfig &cfg, ColorBinaryConfig &color_cfg, bool &foxglove_debug)
{
    std::string used_path;
    try
    {
        YAML::Node root = LoadYoloConfigWithFallback(used_path);
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
        std::cout << "Loaded yolo config: " << used_path << std::endl;
    }
    catch (const std::exception &e)
    {
        cfg.target_color = ResolveTargetColorFromEnv();
        std::cout << "Load yolo config failed, using defaults: " << e.what() << std::endl;
    }
}

static bool IsColorMatched(LightColor detected, const std::string &target_color)
{
    const std::string target = ToLowerCopy(target_color);
    if (target == "red")
    {
        return detected == LightColor::RED;
    }
    if (target == "blue")
    {
        return detected == LightColor::BLUE;
    }
    return false;
}

static ColorBinaryConfig g_color_cfg{};

static LightColor ClassifyLightColorInBoxBinary(const cv::Mat &bgr, const cv::Rect &box)
{
    if (bgr.empty())
    {
        return LightColor::UNKNOWN;
    }

    const cv::Rect image_rect(0, 0, bgr.cols, bgr.rows);
    const cv::Rect roi = box & image_rect;
    if (roi.width < 3 || roi.height < 3)
    {
        return LightColor::UNKNOWN;
    }

    cv::Mat roi_bgr = bgr(roi);

    // Limit per-box workload to protect FPS when detections are many.
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

    const cv::Mat kKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(g_color_cfg.morph_kernel, g_color_cfg.morph_kernel));
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kKernel);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, kKernel);

    const int total_pixels = roi_bgr.cols * roi_bgr.rows;
    if (total_pixels <= 0)
    {
        return LightColor::UNKNOWN;
    }

    const int red_pixels = cv::countNonZero(red_mask);
    const int blue_pixels = cv::countNonZero(blue_mask);
    const float red_ratio = static_cast<float>(red_pixels) / static_cast<float>(total_pixels);
    const float blue_ratio = static_cast<float>(blue_pixels) / static_cast<float>(total_pixels);

    if (red_ratio < g_color_cfg.min_color_ratio && blue_ratio < g_color_cfg.min_color_ratio)
    {
        return LightColor::UNKNOWN;
    }
    if (red_ratio > blue_ratio * g_color_cfg.dominance_ratio)
    {
        return LightColor::RED;
    }
    if (blue_ratio > red_ratio * g_color_cfg.dominance_ratio)
    {
        return LightColor::BLUE;
    }
    return LightColor::UNKNOWN;
}

struct LightBarGeom
{
    cv::Point2f center;
    cv::Point2f top;
    cv::Point2f bottom;
    float area = 0.0f;
};

static bool ComputeLineIntersection(const cv::Point2f &a1,
                                    const cv::Point2f &a2,
                                    const cv::Point2f &b1,
                                    const cv::Point2f &b2,
                                    cv::Point2f &out)
{
    const float x1 = a1.x, y1 = a1.y;
    const float x2 = a2.x, y2 = a2.y;
    const float x3 = b1.x, y3 = b1.y;
    const float x4 = b2.x, y4 = b2.y;
    const float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (std::fabs(denom) < 1e-6f)
    {
        return false;
    }

    const float det1 = x1 * y2 - y1 * x2;
    const float det2 = x3 * y4 - y3 * x4;
    out.x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom;
    out.y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom;
    return true;
}

static bool EstimateCenterByLightBars(const cv::Mat &bgr,
                                      const cv::Rect &box,
                                      LightColor target_color,
                                      cv::Point2f &center,
                                      std::array<cv::Point2f, 4> &cross_lines)
{
    const cv::Rect image_rect(0, 0, bgr.cols, bgr.rows);
    const cv::Rect roi = box & image_rect;
    if (roi.width < 6 || roi.height < 6)
    {
        return false;
    }

    cv::Mat roi_bgr = bgr(roi);
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

    const cv::Mat kKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(g_color_cfg.morph_kernel, g_color_cfg.morph_kernel));
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kKernel);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, kKernel);

    cv::Mat mask = (target_color == LightColor::RED) ? red_mask : blue_mask;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBarGeom> bars;
    bars.reserve(contours.size());
    for (const auto &cnt : contours)
    {
        const float area = static_cast<float>(cv::contourArea(cnt));
        if (area < 8.0f)
        {
            continue;
        }
        const cv::RotatedRect rr = cv::minAreaRect(cnt);
        const float w = std::min(rr.size.width, rr.size.height);
        const float h = std::max(rr.size.width, rr.size.height);
        if (w < 1.0f || h < 4.0f)
        {
            continue;
        }
        if ((h / w) < 1.2f)
        {
            continue;
        }

        cv::Point2f pts[4];
        rr.points(pts);
        std::array<cv::Point2f, 4> p = {pts[0], pts[1], pts[2], pts[3]};
        std::sort(p.begin(), p.end(), [](const cv::Point2f &lhs, const cv::Point2f &rhs) {
            return lhs.y < rhs.y;
        });

        LightBarGeom geom;
        geom.top = (p[0] + p[1]) * 0.5f;
        geom.bottom = (p[2] + p[3]) * 0.5f;
        geom.center = rr.center;
        geom.area = area;

        const cv::Point2f offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
        geom.top += offset;
        geom.bottom += offset;
        geom.center += offset;
        bars.push_back(geom);
    }

    if (bars.size() < 2)
    {
        return false;
    }

    std::sort(bars.begin(), bars.end(), [](const LightBarGeom &a, const LightBarGeom &b) {
        return a.area > b.area;
    });

    LightBarGeom bar_a = bars[0];
    LightBarGeom bar_b = bars[1];
    if (bar_a.center.x > bar_b.center.x)
    {
        std::swap(bar_a, bar_b);
    }

    cross_lines[0] = bar_a.top;
    cross_lines[1] = bar_b.bottom;
    cross_lines[2] = bar_a.bottom;
    cross_lines[3] = bar_b.top;

    if (!ComputeLineIntersection(cross_lines[0], cross_lines[1], cross_lines[2], cross_lines[3], center))
    {
        center = (bar_a.center + bar_b.center) * 0.5f;
    }
    return true;
}

static const char *LightColorShortName(LightColor c)
{
    switch (c)
    {
    case LightColor::RED:
        return "R";
    case LightColor::BLUE:
        return "B";
    default:
        return "U";
    }
}

static bool g_foxglove_debug = false;

static AppConfig g_cfg = []() {
    AppConfig cfg = MakeDefaultAppConfig();
    LoadYoloConfig(cfg, g_color_cfg, g_foxglove_debug);
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
        std::string cmd = "mkdir -p " + kDir;
        std::system(cmd.c_str());
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
};

struct Detection
{
    cv::Rect box;
    int class_id = -1;
    float confidence = 0.0f;
};

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
            auto model = ov::Core().read_model(resolved_model_path);
            compiled = ov::Core().compile_model(model, device);
            request_ = compiled.create_infer_request();
            input_port_ = compiled.input();
            output_port_ = compiled.output();
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
            input_data_.resize(static_cast<size_t>(3 * input_h_ * input_w_));
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

    bool infer(const cv::Mat &frame, cv::Mat &vis, int &det_count, std::string *error)
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
        ov::Tensor in_tensor(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)}, input_data_.data());
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
        for (const auto &d : dets)
        {
            const LightColor light_color = ClassifyLightColorInBoxBinary(frame, d.box);
            if (!IsColorMatched(light_color, g_cfg.target_color))
            {
                continue;
            }
            ++kept_count;

            if (!enable_vis)
            {
                continue;
            }

            const cv::Scalar box_color = (light_color == LightColor::RED)
                                             ? cv::Scalar(0, 0, 255)
                                             : (light_color == LightColor::BLUE ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0));
            cv::rectangle(vis, d.box, box_color, 2);
            std::string text = "id=" + std::to_string(d.class_id) +
                               " conf=" + cv::format("%.2f", d.confidence) +
                               " clr=" + std::string(LightColorShortName(light_color));
            int ty = std::max(0, d.box.y - 5);
            cv::putText(vis, text, cv::Point(d.box.x, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv::LINE_AA);

            cv::Point2f center_pt;
            std::array<cv::Point2f, 4> cross_lines;
            if (EstimateCenterByLightBars(frame, d.box, light_color, center_pt, cross_lines))
            {
                const cv::Scalar cross_color(0, 255, 255);
                cv::line(vis, cross_lines[0], cross_lines[1], cross_color, 2, cv::LINE_AA);
                cv::line(vis, cross_lines[2], cross_lines[3], cross_color, 2, cv::LINE_AA);
                cv::circle(vis, center_pt, 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
            }
        }
        det_count = kept_count;
        return true;
    }

private:
    struct LetterBoxInfo
    {
        float scale = 1.0f;
        int pad_w = 0;
        int pad_h = 0;
    };

    cv::Mat letterbox(const cv::Mat &src, LetterBoxInfo &lb) const
    {
        int src_w = src.cols;
        int src_h = src.rows;
        lb.scale = std::min(static_cast<float>(input_w_) / src_w, static_cast<float>(input_h_) / src_h);
        int nw = static_cast<int>(std::round(src_w * lb.scale));
        int nh = static_cast<int>(std::round(src_h * lb.scale));
        lb.pad_w = (input_w_ - nw) / 2;
        lb.pad_h = (input_h_ - nh) / 2;
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(nw, nh));
        cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(out(cv::Rect(lb.pad_w, lb.pad_h, nw, nh)));
        return out;
    };

    void preprocess(const cv::Mat &bgr)
    {
        cv::Mat lb_img = letterbox(bgr, lb_);
        cv::Mat rgb;
        cv::cvtColor(lb_img, rgb, cv::COLOR_BGR2RGB);
        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3, 1.0f / 255.0f);
        const int H = input_h_;
        const int W = input_w_;
        for (int y = 0; y < H; ++y)
        {
            const cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x)
            {
                input_data_[0 * H * W + y * W + x] = row[x][0];
                input_data_[1 * H * W + y * W + x] = row[x][1];
                input_data_[2 * H * W + y * W + x] = row[x][2];
            }
        }
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
    std::vector<float> input_data_;
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
            CStatus st = CGRAPH_SEND_MPARAM(FrameMParam, FRAME_TOPIC, msg, GMessagePushStrategy::WAIT);
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
            std::string err;
            if (!yolo_.infer(in->frame, vis, det_count, &err))
            {
                return CStatus(err);
            }
            std::shared_ptr<ResultMParam> out(new ResultMParam());
            out->vis = vis;
            out->frame_id = in->frame_id;
            out->infer_id = infer_id;
            out->latency_ms = static_cast<double>(NowMs() - in->ts_ms);
            out->det_count = det_count;
            st = CGRAPH_SEND_MPARAM(ResultMParam, RESULT_TOPIC, out, GMessagePushStrategy::WAIT);
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
            auto now = std::chrono::steady_clock::now();
            auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start_).count();
            if (dt_ms >= 1000) {
                fps_ = count_ * 1000.0 / static_cast<double>(dt_ms);
                count_ = 0;
                fps_start_ = now;
            }

            auto log_dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - log_start_).count();
            if (log_dt_ms >= 1000) {
                double real_fps = log_count_ * 1000.0 / static_cast<double>(log_dt_ms);
                std::cout << "[REAL_FPS] fps=" << static_cast<int>(real_fps + 0.5)
                          << " latency_ms=" << static_cast<int>(r->latency_ms + 0.5)
                          << " det=" << r->det_count
                          << " infer=" << r->infer_id << std::endl;
                log_count_ = 0;
                log_start_ = now;
            }

            if (!enable_vis || r->vis.empty()) {
                continue;
            }

            cv::Mat &show = r->vis;
            std::string t1 = "frame=" + std::to_string(r->frame_id) + " infer=" + std::to_string(r->infer_id) + " det=" + std::to_string(r->det_count);
            std::string t2 = "fps=" + std::to_string(static_cast<int>(fps_ + 0.5));
            std::string t3 = "latency=" + std::to_string(static_cast<int>(r->latency_ms + 0.5)) + "ms";
            cv::putText(show, t1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            cv::putText(show, t2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(show, t3, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
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