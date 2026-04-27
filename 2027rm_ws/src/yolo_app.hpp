#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

#include <CGraph.h>
#include <opencv2/opencv.hpp>

struct AppConfig {
    std::string model_path;
    std::string device;
    int num_classes;
    float conf_thres;
    float nms_thres;
    int fast_nms_topk;
    int thread_num;
    int infer_workers;
    int max_color_candidates;
    bool ema_enable = true;
    double ema_alpha = 0.45;
    uint64_t ema_reset_frame_gap = 10;
    std::string target_color;
    bool pnp_enable = true;
    double pnp_small_armor_width_m = 0.137;
    double pnp_big_armor_width_m = 0.230;
    double pnp_armor_length_m = 0.125;
    double pnp_lightbar_length_m = 0.057;
    bool pnp_rigid_constraint_enable = true;
    double pnp_max_mean_reprojection_error_px = 4.0;
    double pnp_max_corner_reprojection_error_px = 8.0;
    double pnp_min_depth_m = 0.05;
    double pnp_max_depth_m = 20.0;
};

inline constexpr const char *FRAME_TOPIC = "rm/frame/topic";
inline constexpr const char *RESULT_TOPIC = "rm/result/topic";
inline constexpr const char *PNP_TOPIC = "rm/pnp/topic";

struct FrameMParam : public CGraph::GMessageParam
{
    cv::Mat frame;
    uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point capture_tp;
};

struct ResultMParam : public CGraph::GMessageParam
{
    cv::Mat vis;
    uint64_t frame_id = 0;
    int infer_id = 0;
    double latency_ms = 0.0;
    std::chrono::steady_clock::time_point capture_tp;
    int det_count = 0;
    bool has_corners = false;
    std::array<cv::Point2f, 4> corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f)};
};

struct PnpResultMParam : public CGraph::GMessageParam
{
    uint64_t frame_id = 0;
    int infer_id = 0;
    bool has_pose = false;
    std::string status;
    std::string armor_type;
    cv::Point2f center_px = cv::Point2f(0.0f, 0.0f);
    cv::Vec3d tvec_m = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d rvec = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d ypr_deg = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec4d quat_xyzw = cv::Vec4d(0.0, 0.0, 0.0, 1.0);
    cv::Vec2d armor_size_m = cv::Vec2d(0.0, 0.0);
};

const AppConfig& GetAppConfig();
void RegisterPnpPipelineElements(CGraph::GPipeline* const &pipeline,
                                 CGraph::GElementPtr *pnp_ref = nullptr,
                                 const CGraph::GElementPtrSet &depends = {});
CStatus RegisterYoloPipelineElements(CGraph::GPipeline* const &pipeline);
void InitYoloMessageTopics();
void ShutdownYoloApp();
