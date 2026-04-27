#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "yolo_app.hpp"

using namespace CGraph;

#ifndef CAMERA_INFO_PATH
#define CAMERA_INFO_PATH "../config/camera_info.yaml"
#endif

namespace {

enum class ArmorType
{
    Small,
    Large,
};

struct CameraCalibration
{
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
};

struct ArmorGeometry
{
    double small_width_m = 0.137;
    double big_width_m = 0.230;
    double armor_length_m = 0.125;
    double lightbar_length_m = 0.057;
};

struct RigidPnpConstraint
{
    bool enable = true;
    double max_mean_reprojection_error_px = 4.0;
    double max_corner_reprojection_error_px = 8.0;
    double min_depth_m = 0.05;
    double max_depth_m = 20.0;
};

ArmorGeometry LoadArmorGeometry()
{
    const AppConfig &cfg = GetAppConfig();
    ArmorGeometry geometry;
    geometry.small_width_m = std::max(0.001, cfg.pnp_small_armor_width_m);
    geometry.big_width_m = std::max(0.001, cfg.pnp_big_armor_width_m);
    geometry.armor_length_m = std::max(0.001, cfg.pnp_armor_length_m);
    geometry.lightbar_length_m = std::max(0.001, cfg.pnp_lightbar_length_m);
    return geometry;
}

RigidPnpConstraint LoadRigidPnpConstraint()
{
    const AppConfig &cfg = GetAppConfig();
    RigidPnpConstraint constraint;
    constraint.enable = cfg.pnp_rigid_constraint_enable;
    constraint.max_mean_reprojection_error_px = std::max(0.0, cfg.pnp_max_mean_reprojection_error_px);
    constraint.max_corner_reprojection_error_px = std::max(0.0, cfg.pnp_max_corner_reprojection_error_px);
    constraint.min_depth_m = std::max(0.0, cfg.pnp_min_depth_m);
    constraint.max_depth_m = std::max(constraint.min_depth_m, cfg.pnp_max_depth_m);
    return constraint;
}

std::vector<cv::Point3f> BuildArmorPoints(double width_m, double lightbar_length_m)
{
    const float half_w = static_cast<float>(width_m * 0.5);
    const float half_h = static_cast<float>(lightbar_length_m * 0.5);
    return {
        {0.0f, half_w, half_h},
        {0.0f, -half_w, half_h},
        {0.0f, -half_w, -half_h},
        {0.0f, half_w, -half_h}};
}

std::string ResolveCameraInfoPath()
{
    const std::vector<std::string> candidates = {
        CAMERA_INFO_PATH,
        "../config/camera_info.yaml",
        "2027rm_ws/config/camera_info.yaml",
        "config/camera_info.yaml"};

    for (const auto &path : candidates)
    {
        std::ifstream fin(path);
        if (fin.good())
        {
            return path;
        }
    }
    return CAMERA_INFO_PATH;
}

cv::Mat LoadMatrixFromYaml(const YAML::Node &node, int rows, int cols)
{
    const std::vector<double> data = node["data"].as<std::vector<double>>();
    cv::Mat mat(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            mat.at<double>(r, c) = data[static_cast<std::size_t>(r * cols + c)];
        }
    }
    return mat;
}

CameraCalibration LoadCameraCalibration()
{
    const std::string path = ResolveCameraInfoPath();
    YAML::Node root = YAML::LoadFile(path);

    CameraCalibration calib;
    calib.camera_matrix = LoadMatrixFromYaml(root["camera_matrix"], 3, 3);

    const YAML::Node dist = root["distortion_coefficients"];
    const int dist_rows = dist["rows"].as<int>();
    const int dist_cols = dist["cols"].as<int>();
    const std::vector<double> dist_data = dist["data"].as<std::vector<double>>();
    calib.distortion_coeffs = cv::Mat(dist_rows, dist_cols, CV_64F);
    for (int i = 0; i < static_cast<int>(dist_data.size()); ++i)
    {
        calib.distortion_coeffs.at<double>(i / dist_cols, i % dist_cols) = dist_data[static_cast<std::size_t>(i)];
    }

    spdlog::info("Loaded camera calibration: {}", path);
    return calib;
}

float Distance(const cv::Point2f &a, const cv::Point2f &b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

ArmorType InferArmorType(const std::array<cv::Point2f, 4> &corners, const ArmorGeometry &geometry)
{
    const float width_px = 0.5f * (Distance(corners[0], corners[1]) + Distance(corners[2], corners[3]));
    const float height_px = 0.5f * (Distance(corners[0], corners[3]) + Distance(corners[1], corners[2]));
    if (height_px <= 1e-3f)
    {
        return ArmorType::Small;
    }

    const float aspect = width_px / height_px;
    const double small_aspect = geometry.small_width_m / geometry.lightbar_length_m;
    const double large_aspect = geometry.big_width_m / geometry.lightbar_length_m;
    const float split_aspect = 0.5f * (small_aspect + large_aspect);
    return aspect >= split_aspect ? ArmorType::Large : ArmorType::Small;
}

const char *ArmorTypeName(ArmorType armor_type)
{
    return armor_type == ArmorType::Large ? "large" : "small";
}

cv::Vec2d ArmorSizeMeters(ArmorType armor_type, const ArmorGeometry &geometry)
{
    if (armor_type == ArmorType::Large)
    {
        return cv::Vec2d(geometry.big_width_m, geometry.lightbar_length_m);
    }
    return cv::Vec2d(geometry.small_width_m, geometry.lightbar_length_m);
}

cv::Point2f ComputeCenterPoint(const std::array<cv::Point2f, 4> &corners)
{
    cv::Point2f center(0.0f, 0.0f);
    for (const auto &corner : corners)
    {
        center += corner;
    }
    return center * 0.25f;
}

cv::Vec4d RotationMatrixToQuaternion(const cv::Mat &rotation_matrix)
{
    const double m00 = rotation_matrix.at<double>(0, 0);
    const double m01 = rotation_matrix.at<double>(0, 1);
    const double m02 = rotation_matrix.at<double>(0, 2);
    const double m10 = rotation_matrix.at<double>(1, 0);
    const double m11 = rotation_matrix.at<double>(1, 1);
    const double m12 = rotation_matrix.at<double>(1, 2);
    const double m20 = rotation_matrix.at<double>(2, 0);
    const double m21 = rotation_matrix.at<double>(2, 1);
    const double m22 = rotation_matrix.at<double>(2, 2);

    cv::Vec4d q(0.0, 0.0, 0.0, 1.0);
    const double trace = m00 + m11 + m22;
    if (trace > 0.0)
    {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        q[3] = 0.25 / s;
        q[0] = (m21 - m12) * s;
        q[1] = (m02 - m20) * s;
        q[2] = (m10 - m01) * s;
        return q;
    }

    if (m00 > m11 && m00 > m22)
    {
        const double s = 2.0 * std::sqrt(1.0 + m00 - m11 - m22);
        q[3] = (m21 - m12) / s;
        q[0] = 0.25 * s;
        q[1] = (m01 + m10) / s;
        q[2] = (m02 + m20) / s;
        return q;
    }
    if (m11 > m22)
    {
        const double s = 2.0 * std::sqrt(1.0 + m11 - m00 - m22);
        q[3] = (m02 - m20) / s;
        q[0] = (m01 + m10) / s;
        q[1] = 0.25 * s;
        q[2] = (m12 + m21) / s;
        return q;
    }

    const double s = 2.0 * std::sqrt(1.0 + m22 - m00 - m11);
    q[3] = (m10 - m01) / s;
    q[0] = (m02 + m20) / s;
    q[1] = (m12 + m21) / s;
    q[2] = 0.25 * s;
    return q;
}

cv::Vec4d RodriguesToQuaternion(const cv::Mat &rvec)
{
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    return RotationMatrixToQuaternion(rotation_matrix);
}

void DumpPnpForFoxglove(const PnpResultMParam &result)
{
    static const std::string kDir = "/tmp/rm_rerun";
    static const std::string kTmp = kDir + "/pnp_latest.tmp.json";
    static const std::string kOut = kDir + "/pnp_latest.json";
    static bool init = false;
    if (!init)
    {
        std::error_code ec;
        std::filesystem::create_directories(kDir, ec);
        init = true;
    }

    std::ofstream fout(kTmp, std::ios::trunc);
    if (!fout.good())
    {
        return;
    }

    fout << "{\n";
    fout << "  \"frame_id\": " << result.frame_id << ",\n";
    fout << "  \"infer_id\": " << result.infer_id << ",\n";
    fout << "  \"has_pose\": " << (result.has_pose ? "true" : "false") << ",\n";
    fout << "  \"status\": \"" << result.status << "\",\n";
    fout << "  \"armor_type\": \"" << result.armor_type << "\",\n";
    fout << "  \"center_px\": [" << result.center_px.x << ", " << result.center_px.y << "],\n";
    fout << "  \"tvec_m\": [" << result.tvec_m[0] << ", " << result.tvec_m[1] << ", " << result.tvec_m[2] << "],\n";
    fout << "  \"rvec\": [" << result.rvec[0] << ", " << result.rvec[1] << ", " << result.rvec[2] << "],\n";
    fout << "  \"ypr_deg\": [" << result.ypr_deg[0] << ", " << result.ypr_deg[1] << ", " << result.ypr_deg[2] << "],\n";
    fout << "  \"quat_xyzw\": [" << result.quat_xyzw[0] << ", " << result.quat_xyzw[1] << ", " << result.quat_xyzw[2] << ", " << result.quat_xyzw[3] << "],\n";
    fout << "  \"armor_size_m\": [" << result.armor_size_m[0] << ", " << result.armor_size_m[1] << "]\n";
    fout << "}\n";
    fout.close();
    std::rename(kTmp.c_str(), kOut.c_str());
}

cv::Vec3d RotationMatrixToEulerZyxDeg(const cv::Mat &rotation_matrix)
{
    const double r00 = rotation_matrix.at<double>(0, 0);
    const double r10 = rotation_matrix.at<double>(1, 0);
    const double r20 = rotation_matrix.at<double>(2, 0);
    const double r21 = rotation_matrix.at<double>(2, 1);
    const double r22 = rotation_matrix.at<double>(2, 2);

    const double yaw = std::atan2(r10, r00);
    const double pitch = std::atan2(-r20, std::sqrt(r21 * r21 + r22 * r22));
    const double roll = std::atan2(r21, r22);

    constexpr double kRadToDeg = 180.0 / CV_PI;
    return cv::Vec3d(yaw * kRadToDeg, pitch * kRadToDeg, roll * kRadToDeg);
}

double ArmorNormalForwardScore(const cv::Mat &rvec)
{
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    const double normal_x = rotation_matrix.at<double>(0, 0);
    const double normal_z = rotation_matrix.at<double>(2, 0);
    return std::abs(std::atan2(normal_x, normal_z));
}

struct ReprojectionStats
{
    double mean_error_px = std::numeric_limits<double>::max();
    double max_error_px = std::numeric_limits<double>::max();
};

ReprojectionStats ComputeReprojectionStats(const std::vector<cv::Point3f> &object_corners,
                                           const std::vector<cv::Point2f> &image_corners,
                                           const CameraCalibration &calibration,
                                           const cv::Mat &rvec,
                                           const cv::Mat &tvec)
{
    std::vector<cv::Point2f> projected_corners;
    cv::projectPoints(object_corners,
                      rvec,
                      tvec,
                      calibration.camera_matrix,
                      calibration.distortion_coeffs,
                      projected_corners);

    ReprojectionStats stats;
    stats.mean_error_px = 0.0;
    stats.max_error_px = 0.0;
    for (std::size_t i = 0; i < image_corners.size() && i < projected_corners.size(); ++i)
    {
        const double error = Distance(image_corners[i], projected_corners[i]);
        stats.mean_error_px += error;
        stats.max_error_px = std::max(stats.max_error_px, error);
    }
    if (!image_corners.empty())
    {
        stats.mean_error_px /= static_cast<double>(image_corners.size());
    }
    return stats;
}

bool HasValidRigidDepth(const std::vector<cv::Point3f> &object_corners,
                        const cv::Mat &rvec,
                        const cv::Mat &tvec,
                        const RigidPnpConstraint &constraint)
{
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    for (const auto &corner : object_corners)
    {
        cv::Mat object_point = (cv::Mat_<double>(3, 1) << corner.x, corner.y, corner.z);
        cv::Mat camera_point = rotation_matrix * object_point + tvec;
        const double depth_m = camera_point.at<double>(2, 0);
        if (depth_m < constraint.min_depth_m || depth_m > constraint.max_depth_m)
        {
            return false;
        }
    }
    return true;
}

std::string FormatReprojectionStatus(const std::string &prefix, const ReprojectionStats &stats)
{
    return prefix + " mean=" + cv::format("%.1f", stats.mean_error_px) +
           " max=" + cv::format("%.1f", stats.max_error_px);
}

bool SatisfiesRigidConstraint(const std::vector<cv::Point3f> &object_corners,
                              const std::vector<cv::Point2f> &image_corners,
                              const CameraCalibration &calibration,
                              const cv::Mat &rvec,
                              const cv::Mat &tvec,
                              const RigidPnpConstraint &constraint,
                              ReprojectionStats *stats_out = nullptr)
{
    const ReprojectionStats stats = ComputeReprojectionStats(object_corners,
                                                             image_corners,
                                                             calibration,
                                                             rvec,
                                                             tvec);
    if (stats_out)
    {
        *stats_out = stats;
    }
    if (!constraint.enable)
    {
        return true;
    }
    if (!HasValidRigidDepth(object_corners, rvec, tvec, constraint))
    {
        return false;
    }
    return stats.mean_error_px <= constraint.max_mean_reprojection_error_px &&
           stats.max_error_px <= constraint.max_corner_reprojection_error_px;
}

bool SolveArmorPnpIppe(const std::vector<cv::Point3f> &object_corners,
                       const std::vector<cv::Point2f> &image_corners,
                       const CameraCalibration &calibration,
                       const RigidPnpConstraint &constraint,
                       cv::Mat &best_rvec,
                       cv::Mat &best_tvec)
{
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    std::vector<double> reprojection_errors;
    const int solution_count = cv::solvePnPGeneric(object_corners,
                                                   image_corners,
                                                   calibration.camera_matrix,
                                                   calibration.distortion_coeffs,
                                                   rvecs,
                                                   tvecs,
                                                   false,
                                                   cv::SOLVEPNP_IPPE,
                                                   cv::noArray(),
                                                   cv::noArray(),
                                                   reprojection_errors);
    if (solution_count <= 0 || rvecs.empty() || tvecs.empty())
    {
        return false;
    }

    std::size_t best_index = 0;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < rvecs.size(); ++i)
    {
        if (constraint.enable && !HasValidRigidDepth(object_corners, rvecs[i], tvecs[i], constraint))
        {
            continue;
        }
        const ReprojectionStats stats = ComputeReprojectionStats(object_corners,
                                                                 image_corners,
                                                                 calibration,
                                                                 rvecs[i],
                                                                 tvecs[i]);
        const double reprojection_score =
            (i < reprojection_errors.size()) ? reprojection_errors[i] : stats.mean_error_px;
        const double forward_score = ArmorNormalForwardScore(rvecs[i]);
        const double score = reprojection_score + forward_score * 10.0;
        if (score < best_score)
        {
            best_score = score;
            best_index = i;
        }
    }

    if (best_score == std::numeric_limits<double>::max())
    {
        return false;
    }

    best_rvec = rvecs[best_index].clone();
    best_tvec = tvecs[best_index].clone();
    return true;
}

class PnpSolveNode : public GNode
{
public:
    CStatus init() override
    {
        result_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(ResultMParam, RESULT_TOPIC, 2);
        if (!GetAppConfig().pnp_enable)
        {
            enabled_ = false;
            spdlog::info("PnP disabled by config");
            return CStatus();
        }

        try
        {
            calibration_ = LoadCameraCalibration();
            geometry_ = LoadArmorGeometry();
            constraint_ = LoadRigidPnpConstraint();
            big_armor_points_ = BuildArmorPoints(geometry_.big_width_m, geometry_.lightbar_length_m);
            small_armor_points_ = BuildArmorPoints(geometry_.small_width_m, geometry_.lightbar_length_m);
            enabled_ = true;
            spdlog::info("PnP armor size loaded: small={:.1f}x{:.1f}mm big={:.1f}x{:.1f}mm lightbar={:.1f}mm rigid_constraint={} reproj_mean<={:.1f}px reproj_corner<={:.1f}px depth=[{:.2f},{:.2f}]m",
                         geometry_.small_width_m * 1000.0,
                         geometry_.armor_length_m * 1000.0,
                         geometry_.big_width_m * 1000.0,
                         geometry_.armor_length_m * 1000.0,
                         geometry_.lightbar_length_m * 1000.0,
                         constraint_.enable,
                         constraint_.max_mean_reprojection_error_px,
                         constraint_.max_corner_reprojection_error_px,
                         constraint_.min_depth_m,
                         constraint_.max_depth_m);
            return CStatus();
        }
        catch (const std::exception &e)
        {
            enabled_ = false;
            return CStatus(std::string("load camera calibration failed: ") + e.what());
        }
    }

    CStatus run() override
    {
        if (!enabled_)
        {
            return CStatus();
        }

        auto stat_start = std::chrono::steady_clock::now();
        int recv_count = 0;
        int no_corner_count = 0;
        int solved_count = 0;
        int reject_count = 0;
        int fail_count = 0;
        auto maybe_log_stats = [&]() {
            const auto now = std::chrono::steady_clock::now();
            const auto stat_dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - stat_start).count();
            if (stat_dt_ms < 1000)
            {
                return;
            }
            spdlog::info("[PNP] recv={} solved={} rejected={} failed={} no_corners={}",
                         recv_count,
                         solved_count,
                         reject_count,
                         fail_count,
                         no_corner_count);
            recv_count = 0;
            solved_count = 0;
            reject_count = 0;
            fail_count = 0;
            no_corner_count = 0;
            stat_start = now;
        };

        while (true)
        {
            std::shared_ptr<ResultMParam> result = nullptr;
            const CStatus st = CGRAPH_SUB_MPARAM_WITH_TIMEOUT(ResultMParam, result_conn_id_, result, 1000);
            if (st.isErr())
            {
                maybe_log_stats();
                continue;
            }
            if (!result || !result->has_corners)
            {
                ++no_corner_count;
                if (result)
                {
                    PublishNoPose(result->frame_id, result->infer_id, "no_corners");
                }
                maybe_log_stats();
                continue;
            }
            ++recv_count;
            if (result->frame_id <= last_processed_frame_id_)
            {
                spdlog::debug("[PNP] drop stale result frame={} latest={}",
                              result->frame_id,
                              last_processed_frame_id_);
                maybe_log_stats();
                continue;
            }
            last_processed_frame_id_ = result->frame_id;

            const std::vector<cv::Point2f> image_corners(result->corners.begin(), result->corners.end());
            const ArmorType armor_type = InferArmorType(result->corners, geometry_);
            const std::vector<cv::Point3f> &object_corners =
                (armor_type == ArmorType::Large)
                    ? big_armor_points_
                    : small_armor_points_;
            const cv::Point2f center_pt = ComputeCenterPoint(result->corners);
            const cv::Vec2d armor_size_m = ArmorSizeMeters(armor_type, geometry_);
            cv::Mat rvec;
            cv::Mat tvec;

            bool ok = SolveArmorPnpIppe(object_corners,
                                        image_corners,
                                        calibration_,
                                        constraint_,
                                        rvec,
                                        tvec);
            if (!ok)
            {
                ok = cv::solvePnP(object_corners,
                                  image_corners,
                                  calibration_.camera_matrix,
                                  calibration_.distortion_coeffs,
                                  rvec,
                                  tvec,
                                  false,
                                  cv::SOLVEPNP_ITERATIVE);
            }
            ReprojectionStats reprojection_stats;
            bool valid_depth = false;
            bool valid_reprojection = false;
            if (ok)
            {
                reprojection_stats = ComputeReprojectionStats(object_corners,
                                                              image_corners,
                                                              calibration_,
                                                              rvec,
                                                              tvec);
                valid_depth = HasValidRigidDepth(object_corners, rvec, tvec, constraint_);
                valid_reprojection =
                    reprojection_stats.mean_error_px <= constraint_.max_mean_reprojection_error_px &&
                    reprojection_stats.max_error_px <= constraint_.max_corner_reprojection_error_px;
            }
            if (ok && constraint_.enable && !valid_depth)
            {
                spdlog::warn("[PNP] depth rejected frame={} infer={} mean_reproj={:.2f}px max_reproj={:.2f}px depth_range=[{:.2f},{:.2f}]m",
                             result->frame_id,
                             result->infer_id,
                             reprojection_stats.mean_error_px,
                             reprojection_stats.max_error_px,
                             constraint_.min_depth_m,
                             constraint_.max_depth_m);
                ++reject_count;
                PublishNoPose(result->frame_id, result->infer_id, FormatReprojectionStatus("depth_rejected", reprojection_stats));
                maybe_log_stats();
                continue;
            }
            if (!ok)
            {
                spdlog::warn("[PNP] solve failed frame={} infer={}", result->frame_id, result->infer_id);
                ++fail_count;
                PublishNoPose(result->frame_id, result->infer_id, "failed");
                maybe_log_stats();
                continue;
            }

            cv::Mat rotation_matrix;
            cv::Rodrigues(rvec, rotation_matrix);
            const cv::Vec3d euler_deg = RotationMatrixToEulerZyxDeg(rotation_matrix);

            std::shared_ptr<PnpResultMParam> pnp_result(new PnpResultMParam());
            pnp_result->frame_id = result->frame_id;
            pnp_result->infer_id = result->infer_id;
            pnp_result->has_pose = true;
            pnp_result->status = FormatReprojectionStatus(valid_reprojection ? "solved" : "loose", reprojection_stats);
            pnp_result->armor_type = ArmorTypeName(armor_type);
            pnp_result->center_px = center_pt;
            pnp_result->tvec_m = cv::Vec3d(
                tvec.at<double>(0, 0),
                tvec.at<double>(1, 0),
                tvec.at<double>(2, 0));
            pnp_result->rvec = cv::Vec3d(
                rvec.at<double>(0, 0),
                rvec.at<double>(1, 0),
                rvec.at<double>(2, 0));
            pnp_result->ypr_deg = euler_deg;
            pnp_result->quat_xyzw = RotationMatrixToQuaternion(rotation_matrix);
            pnp_result->armor_size_m = armor_size_m;
            const CStatus pub_st = CGRAPH_PUB_MPARAM(PnpResultMParam, PNP_TOPIC, pnp_result, GMessagePushStrategy::REPLACE);
            if (pub_st.isErr())
            {
                spdlog::warn("[PNP] publish failed frame={} infer={} error={}",
                             result->frame_id,
                             result->infer_id,
                             pub_st.getInfo());
            }
            DumpPnpForFoxglove(*pnp_result);
            ++solved_count;
            maybe_log_stats();
        }

        return CStatus();
    }

private:
    void PublishNoPose(uint64_t frame_id, int infer_id, const std::string &status)
    {
        std::shared_ptr<PnpResultMParam> pnp_result(new PnpResultMParam());
        pnp_result->frame_id = frame_id;
        pnp_result->infer_id = infer_id;
        pnp_result->has_pose = false;
        pnp_result->status = status;
        const CStatus pub_st = CGRAPH_PUB_MPARAM(PnpResultMParam, PNP_TOPIC, pnp_result, GMessagePushStrategy::REPLACE);
        if (pub_st.isErr())
        {
            spdlog::warn("[PNP] publish no-pose failed frame={} infer={} status={} error={}",
                         frame_id,
                         infer_id,
                         status,
                         pub_st.getInfo());
        }
        DumpPnpForFoxglove(*pnp_result);
    }

    int result_conn_id_ = -1;
    uint64_t last_processed_frame_id_ = 0;
    bool enabled_ = false;
    CameraCalibration calibration_{};
    ArmorGeometry geometry_{};
    RigidPnpConstraint constraint_{};
    std::vector<cv::Point3f> big_armor_points_;
    std::vector<cv::Point3f> small_armor_points_;
};

}  // namespace

void RegisterPnpPipelineElements(CGraph::GPipeline* const &pipeline,
                                 CGraph::GElementPtr *pnp_ref,
                                 const CGraph::GElementPtrSet &depends)
{
    GElementPtr pnp = nullptr;
    CStatus st = pipeline->registerGElement<PnpSolveNode>(&pnp, depends, "PnP 解算\npnp_solver");
    if (st.isErr())
    {
        spdlog::error("register pnp failed: {}", st.getInfo());
    }
    if (pnp_ref)
    {
        *pnp_ref = pnp;
    }
}
