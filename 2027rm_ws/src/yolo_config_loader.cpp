#include "yolo_config_loader.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "yolo_common.hpp"

#ifndef YOLO_CONFIG_PATH
#define YOLO_CONFIG_PATH "../config/yolo_config.yaml"
#endif

namespace {

bool g_foxglove_debug = false;
app::logging::LogConfig g_log_cfg{};
bool g_e2e_log_enable = true;
std::atomic<bool> g_stop(false);

std::string ResolveTargetColorFromEnv()
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

YAML::Node LoadYoloConfigWithFallback(std::string &used_path)
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

void LoadLoggingConfigNode(const YAML::Node &logging, app::logging::LogConfig &log_cfg)
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

void LoadPerfLogSwitchNode(const YAML::Node &logging, bool &e2e_log_enable)
{
    if (!logging)
    {
        return;
    }

    if (logging["e2e_enable"]) e2e_log_enable = logging["e2e_enable"].as<bool>();
}

app::logging::LogConfig LoadEarlyLoggingConfig()
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

void LoadYoloConfig(AppConfig &cfg)
{
    std::string used_path;
    try
    {
        YAML::Node root = LoadYoloConfigWithFallback(used_path);
        LoadLoggingConfigNode(root["logging"], g_log_cfg);
        LoadPerfLogSwitchNode(root["logging"], g_e2e_log_enable);
        app::logging::ApplyRuntimeLoggingConfig(g_log_cfg);

        const YAML::Node yolo = root["yolo"];
        if (yolo)
        {
            if (yolo["model_path"]) cfg.model_path = yolo["model_path"].as<std::string>();
            if (yolo["device"]) cfg.device = yolo["device"].as<std::string>();
            if (yolo["num_classes"]) cfg.num_classes = yolo["num_classes"].as<int>();
            if (yolo["conf_thres"]) cfg.conf_thres = yolo["conf_thres"].as<float>();
            if (yolo["nms_thres"]) cfg.nms_thres = yolo["nms_thres"].as<float>();
            if (yolo["fast_nms_topk"]) cfg.fast_nms_topk = yolo["fast_nms_topk"].as<int>();
            if (yolo["thread_num"]) cfg.thread_num = std::max(2, yolo["thread_num"].as<int>());
            if (yolo["infer_workers"]) cfg.infer_workers = std::max(1, yolo["infer_workers"].as<int>());
            if (yolo["max_color_candidates"]) cfg.max_color_candidates = std::max(1, yolo["max_color_candidates"].as<int>());
            if (yolo["ema_enable"]) cfg.ema_enable = yolo["ema_enable"].as<bool>();
            if (yolo["ema_alpha"]) cfg.ema_alpha = std::clamp(yolo["ema_alpha"].as<double>(), 0.0, 1.0);
            if (yolo["ema_reset_frame_gap"]) cfg.ema_reset_frame_gap = std::max<uint64_t>(1, yolo["ema_reset_frame_gap"].as<uint64_t>());
            if (yolo["target_color"]) cfg.target_color = ToLowerCopy(yolo["target_color"].as<std::string>());
        }

        const YAML::Node pnp = root["pnp"];
        if (pnp)
        {
            if (pnp["enable"]) cfg.pnp_enable = pnp["enable"].as<bool>();
            if (pnp["small_armor_width_m"]) cfg.pnp_small_armor_width_m = std::max(0.001, pnp["small_armor_width_m"].as<double>());
            if (pnp["big_armor_width_m"]) cfg.pnp_big_armor_width_m = std::max(0.001, pnp["big_armor_width_m"].as<double>());
            if (pnp["armor_length_m"]) cfg.pnp_armor_length_m = std::max(0.001, pnp["armor_length_m"].as<double>());
            if (pnp["lightbar_length_m"]) cfg.pnp_lightbar_length_m = std::max(0.001, pnp["lightbar_length_m"].as<double>());
            if (pnp["rigid_constraint_enable"]) cfg.pnp_rigid_constraint_enable = pnp["rigid_constraint_enable"].as<bool>();
            if (pnp["max_mean_reprojection_error_px"]) cfg.pnp_max_mean_reprojection_error_px = std::max(0.0, pnp["max_mean_reprojection_error_px"].as<double>());
            if (pnp["max_corner_reprojection_error_px"]) cfg.pnp_max_corner_reprojection_error_px = std::max(0.0, pnp["max_corner_reprojection_error_px"].as<double>());
            if (pnp["min_depth_m"]) cfg.pnp_min_depth_m = std::max(0.0, pnp["min_depth_m"].as<double>());
            if (pnp["max_depth_m"]) cfg.pnp_max_depth_m = std::max(cfg.pnp_min_depth_m, pnp["max_depth_m"].as<double>());
        }

        const YAML::Node debug = root["debug"];
        if (debug && debug["foxglove_enable"])
        {
            g_foxglove_debug = debug["foxglove_enable"].as<bool>();
        }

        const YAML::Node profiling = root["profiling"];
        if (profiling)
        {
            if (profiling["enable"]) cfg.profiling_enable = profiling["enable"].as<bool>();
            if (profiling["interval_ms"]) cfg.profiling_interval_ms = std::max(100, profiling["interval_ms"].as<int>());
        }

        if (cfg.target_color != "red" && cfg.target_color != "blue")
        {
            cfg.target_color = ResolveTargetColorFromEnv();
        }
        const int min_pipeline_threads = cfg.pnp_enable ? 4 : 3;
        cfg.thread_num = std::max(cfg.thread_num, min_pipeline_threads);
        spdlog::info("Loaded yolo config: {}", used_path);
    }
    catch (const std::exception &e)
    {
        cfg.target_color = ResolveTargetColorFromEnv();
        const int min_pipeline_threads = cfg.pnp_enable ? 4 : 3;
        cfg.thread_num = std::max(cfg.thread_num, min_pipeline_threads);
        spdlog::warn("Load yolo config failed, using defaults: {}", e.what());
    }
}

}  // namespace

AppConfig MakeDefaultAppConfig()
{
    return AppConfig{
        "model/last.xml",
        "CPU",
        0,
        0.25f,
        0.25f,
        200,
        4,
        2,
        8,
        true,
        0.45,
        10,
        "blue",
        true,
        0.135,
        0.230,
        0.056};
}

void InitYoloRuntimeConfig()
{
    const app::logging::LogConfig early_log_cfg = LoadEarlyLoggingConfig();
    app::logging::InitAsyncLogging(early_log_cfg);
}

const app::logging::LogConfig &GetYoloLogConfig()
{
    return g_log_cfg;
}

std::string ToLowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s;
}

TargetColor ParseTargetColor(const std::string &color)
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

bool IsFoxgloveDebugEnabled()
{
    return g_foxglove_debug;
}

bool IsE2eLogEnabled()
{
    return g_e2e_log_enable;
}

bool IsYoloStopRequested()
{
    return g_stop.load();
}

void RequestYoloStop()
{
    g_stop.store(true);
}

void ClearYoloMessages()
{
    CGRAPH_CLEAR_MESSAGES()
}

double ElapsedMsSince(const std::chrono::steady_clock::time_point &start_tp)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_tp).count();
}

void DumpFrameForFoxglove(const cv::Mat &img)
{
    if (!IsFoxgloveDebugEnabled() || img.empty())
    {
        return;
    }

    static const std::string kDir = "/tmp/rm_rerun";
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
    const std::string tmp = kDir + "/latest." +
                            std::to_string(static_cast<unsigned long long>(
                                std::chrono::steady_clock::now().time_since_epoch().count())) +
                            ".tmp.jpg";
    if (!cv::imwrite(tmp, img, params))
    {
        spdlog::warn("write debug frame failed: {}", tmp);
        return;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, kOut, ec);
    if (ec)
    {
        std::filesystem::remove(kOut, ec);
        ec.clear();
        std::filesystem::rename(tmp, kOut, ec);
    }
    if (ec)
    {
        spdlog::warn("publish debug frame rename failed: {} -> {} error={}", tmp, kOut, ec.message());
        std::filesystem::remove(tmp, ec);
    }
}

std::array<cv::Point2f, 4> OrderCorners(const std::array<cv::Point2f, 4> &points)
{
    std::array<cv::Point2f, 4> ordered;
    float min_sum = std::numeric_limits<float>::max();
    float max_sum = -std::numeric_limits<float>::max();
    float min_diff = std::numeric_limits<float>::max();
    float max_diff = -std::numeric_limits<float>::max();
    for (const auto &pt : points)
    {
        const float sum = pt.x + pt.y;
        const float diff = pt.x - pt.y;
        if (sum < min_sum)
        {
            min_sum = sum;
            ordered[0] = pt;
        }
        if (diff > max_diff)
        {
            max_diff = diff;
            ordered[1] = pt;
        }
        if (sum > max_sum)
        {
            max_sum = sum;
            ordered[2] = pt;
        }
        if (diff < min_diff)
        {
            min_diff = diff;
            ordered[3] = pt;
        }
    }
    return ordered;
}

void DrawArmorFrame(cv::Mat &vis,
                    const std::array<cv::Point2f, 4> &corners,
                    const cv::Scalar &accent_color)
{
    for (std::size_t i = 0; i < corners.size(); ++i)
    {
        const auto &a = corners[i];
        const auto &b = corners[(i + 1) % corners.size()];
        cv::line(vis, a, b, cv::Scalar(0, 0, 0), 6, cv::LINE_AA);
        cv::line(vis, a, b, accent_color, 3, cv::LINE_AA);
    }

    cv::line(vis, corners[0], corners[2], cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
    cv::line(vis, corners[1], corners[3], cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
    cv::line(vis, corners[0], corners[2], cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::line(vis, corners[1], corners[3], cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::line(vis, corners[0], corners[2], accent_color, 1, cv::LINE_AA);
    cv::line(vis, corners[1], corners[3], accent_color, 1, cv::LINE_AA);

    cv::Point2f center(0.0f, 0.0f);
    for (const auto &corner : corners)
    {
        center += corner;
    }
    center *= 0.25f;
    cv::drawMarker(vis, center, cv::Scalar(0, 0, 0), cv::MARKER_CROSS, 26, 5, cv::LINE_AA);
    cv::drawMarker(vis, center, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 22, 2, cv::LINE_AA);
    cv::drawMarker(vis, center, accent_color, cv::MARKER_CROSS, 16, 1, cv::LINE_AA);

    for (std::size_t i = 0; i < corners.size(); ++i)
    {
        const auto &corner = corners[i];
        cv::circle(vis, corner, 7, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
        cv::circle(vis, corner, 4, accent_color, -1, cv::LINE_AA);
        cv::putText(vis, std::to_string(i), corner + cv::Point2f(6.0f, -6.0f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(vis, std::to_string(i), corner + cv::Point2f(6.0f, -6.0f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

std::string ResolveModelPath(const std::string &model_path)
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

std::vector<std::string> LoadModelLabels(const std::string &model_path)
{
    std::ifstream fin(model_path);
    if (!fin.good())
    {
        return {};
    }

    std::stringstream buffer;
    buffer << fin.rdbuf();
    const std::string xml = buffer.str();
    const std::string key = "<labels value=\"";
    const std::size_t start = xml.find(key);
    if (start == std::string::npos)
    {
        return {};
    }

    const std::size_t value_begin = start + key.size();
    const std::size_t value_end = xml.find('"', value_begin);
    if (value_end == std::string::npos || value_end <= value_begin)
    {
        return {};
    }

    std::vector<std::string> labels;
    std::istringstream iss(xml.substr(value_begin, value_end - value_begin));
    std::string label;
    while (iss >> label)
    {
        labels.push_back(ToLowerCopy(label));
    }
    return labels;
}

static AppConfig g_cfg = []() {
    InitYoloRuntimeConfig();
    AppConfig cfg = MakeDefaultAppConfig();
    LoadYoloConfig(cfg);
    return cfg;
}();

const AppConfig &GetAppConfig()
{
    return g_cfg;
}
