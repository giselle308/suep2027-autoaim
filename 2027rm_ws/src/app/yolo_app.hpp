#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>

#include "memory_layout.hpp"

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
    bool armor_classifier_enable = true;
    std::string armor_classifier_model_path = "model/tiny_resnet.onnx";
    double armor_classifier_confidence = 0.70;
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
    bool pnp_yaw_refine_enable = true;
    double pnp_yaw_search_range_deg = 45.0;
    double pnp_yaw_search_step_deg = 1.0;
    bool profiling_enable = false;
    int profiling_interval_ms = 1000;
    int frame_buffer_pool_size = 0;
    int frame_message_pool_size = 0;
    int result_message_pool_size = 0;
    int pnp_message_pool_size = 0;
    int rgo_message_pool_size = 0;
    int imu_message_pool_size = 0;
    bool affinity_enable = false;
    int camera_cpu = -1;
    int infer_cpu = -1;
    int pnp_cpu = -1;
    int rgo_cpu = -1;
    int display_cpu = -1;
    int serial_cpu = -1;
    bool serial_enable = false;
    std::string serial_device = "/dev/ttyACM0";
    int serial_baud_rate = 921600;
};

inline constexpr const char *FRAME_TOPIC = "rm/frame/topic";
inline constexpr const char *RESULT_TOPIC = "rm/result/topic";
inline constexpr const char *PNP_TOPIC = "rm/pnp/topic";
inline constexpr const char *RGO_OUTPUT_TOPIC = "rm/rgo/output";
inline constexpr const char *IMU_TOPIC = "rm/imu/topic";

struct alignas(app::memory::kCacheLineSize) FrameMParam : public CGraph::GMessageParam
{
    cv::Mat frame;
    uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point pipeline_start_tp;
    std::chrono::steady_clock::time_point capture_tp;
};

struct alignas(app::memory::kCacheLineSize) ResultMParam : public CGraph::GMessageParam
{
    struct ArmorDetection
    {
        cv::Rect box;
        int class_id = -1;
        float confidence = 0.0f;
        int armor_name_id = -1;
        float armor_name_confidence = 0.0f;
        std::string armor_name;
        std::array<cv::Point2f, 4> corners = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f)};
    };

    cv::Mat vis;
    uint64_t frame_id = 0;
    int infer_id = 0;
    double latency_ms = 0.0;
    double detect_latency_ms = 0.0;
    std::chrono::steady_clock::time_point pipeline_start_tp;
    std::chrono::steady_clock::time_point capture_tp;
    int det_count = 0;
    int class_id = -1;
    int armor_name_id = -1;
    float armor_name_confidence = 0.0f;
    std::string armor_name;
    bool has_corners = false;
    std::array<cv::Point2f, 4> corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(0.0f, 0.0f)};
    std::vector<ArmorDetection> armors;
};

struct alignas(app::memory::kCacheLineSize) PnpResultMParam : public CGraph::GMessageParam
{
    struct ArmorPose
    {
        bool has_pose = false;
        std::string armor_type;
        int class_id = -1;
        int armor_name_id = -1;
        float armor_name_confidence = 0.0f;
        std::string armor_name;
        double mean_reprojection_error_px = 0.0;
        double max_reprojection_error_px = 0.0;
        bool reprojection_ok = false;
        bool depth_ok = false;
        cv::Point2f center_px = cv::Point2f(0.0f, 0.0f);
        cv::Vec3d tvec_m = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d rvec = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d ypr_deg = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec4d quat_xyzw = cv::Vec4d(0.0, 0.0, 0.0, 1.0);
        cv::Vec2d armor_size_m = cv::Vec2d(0.0, 0.0);
    };

    uint64_t frame_id = 0;
    int infer_id = 0;
    bool has_pose = false;
    double latency_ms = 0.0;
    double detect_latency_ms = 0.0;
    std::string status;
    std::string armor_type;
    int class_id = -1;
    int armor_name_id = -1;
    float armor_name_confidence = 0.0f;
    std::string armor_name;
    double mean_reprojection_error_px = 0.0;
    double max_reprojection_error_px = 0.0;
    bool reprojection_ok = false;
    bool depth_ok = false;
    cv::Point2f center_px = cv::Point2f(0.0f, 0.0f);
    cv::Vec3d tvec_m = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d rvec = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d ypr_deg = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec4d quat_xyzw = cv::Vec4d(0.0, 0.0, 0.0, 1.0);
    cv::Vec2d armor_size_m = cv::Vec2d(0.0, 0.0);
    std::vector<ArmorPose> armors;
};

struct alignas(app::memory::kCacheLineSize) RgoOutputMParam : public CGraph::GMessageParam
{
    uint64_t frame_id = 0;
    int infer_id = 0;
    bool has_target = false;
    double latency_ms = 0.0;
    double detect_latency_ms = 0.0;
    std::string status;
    std::string armor_type;
    int class_id = -1;
    int armor_name_id = -1;
    float armor_name_confidence = 0.0f;
    std::string armor_name;
    double mean_reprojection_error_px = 0.0;
    double max_reprojection_error_px = 0.0;
    bool reprojection_ok = false;
    bool depth_ok = false;
    cv::Vec3d xyz_m = cv::Vec3d(0.0, 0.0, 0.0);
    double yaw_deg = 0.0;
    double bearing_yaw_deg = 0.0;
};

struct alignas(app::memory::kCacheLineSize) ImuMParam : public CGraph::GMessageParam
{
    uint32_t time_stamp_ms = 0;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point receive_tp;
    double yaw_rad = 0.0;
    double pitch_rad = 0.0;
    double roll_rad = 0.0;
    double yaw_vel_rad_s = 0.0;
    double pitch_vel_rad_s = 0.0;
    double roll_vel_rad_s = 0.0;
};

const AppConfig& GetAppConfig();
void RegisterPnpPipelineElements(CGraph::GPipeline* const &pipeline,
                                 CGraph::GElementPtr *pnp_ref = nullptr,
                                 const CGraph::GElementPtrSet &depends = {});
void RegisterRgoPipelineElements(CGraph::GPipeline* const &pipeline,
                                 CGraph::GElementPtr *rgo_ref = nullptr,
                                 const CGraph::GElementPtrSet &depends = {});
CStatus RegisterYoloPipelineElements(CGraph::GPipeline* const &pipeline);
void RegisterSerialImuPipelineElements(CGraph::GPipeline* const &pipeline,
                                       CGraph::GElementPtr *serial_ref = nullptr,
                                       const CGraph::GElementPtrSet &depends = {});
void InitYoloMessageTopics();
void ShutdownYoloApp();
