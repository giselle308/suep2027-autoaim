#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <CGraph.h>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

#include "message_pool.hpp"
#include "thread_affinity.hpp"
#include "yolo_app.hpp"
#include "yolo_common.hpp"

using namespace CGraph;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;

enum class ArmorSlot
{
    FRONT = 0,
    LEFT = 1,
    BACK = 2,
    RIGHT = 3,
};

struct BodyCandidate
{
    ArmorSlot slot = ArmorSlot::FRONT;
    cv::Vec3d center_m = cv::Vec3d(0.0, 0.0, 0.0);
    double yaw_rad = 0.0;
    double score = 0.0;
    double weight = 0.0;
};

struct RgoArmor
{
    ArmorSlot slot = ArmorSlot::FRONT;
    cv::Vec3d center_m = cv::Vec3d(0.0, 0.0, 0.0);
    double yaw_rad = 0.0;
};

struct RgoGeometry
{
    double front_radius_m = 0.22;
    double back_radius_m = 0.22;
    double left_radius_m = 0.20;
    double right_radius_m = 0.20;
};

const char *SlotName(ArmorSlot slot)
{
    switch (slot)
    {
    case ArmorSlot::FRONT:
        return "FRONT";
    case ArmorSlot::LEFT:
        return "LEFT";
    case ArmorSlot::BACK:
        return "BACK";
    case ArmorSlot::RIGHT:
        return "RIGHT";
    }
    return "UNKNOWN";
}

double SlotYawOffsetRad(ArmorSlot slot)
{
    switch (slot)
    {
    case ArmorSlot::FRONT:
        return 0.0;
    case ArmorSlot::LEFT:
        return kPi * 0.5;
    case ArmorSlot::BACK:
        return kPi;
    case ArmorSlot::RIGHT:
        return -kPi * 0.5;
    }
    return 0.0;
}

cv::Vec3d SlotOffsetBody(ArmorSlot slot, const RgoGeometry &g)
{
    switch (slot)
    {
    case ArmorSlot::FRONT:
        return cv::Vec3d(0.0, 0.0, -g.front_radius_m);
    case ArmorSlot::BACK:
        return cv::Vec3d(0.0, 0.0, g.back_radius_m);
    case ArmorSlot::LEFT:
        return cv::Vec3d(-g.left_radius_m, 0.0, 0.0);
    case ArmorSlot::RIGHT:
        return cv::Vec3d(g.right_radius_m, 0.0, 0.0);
    }
    return cv::Vec3d(0.0, 0.0, 0.0);
}

cv::Vec3d RotateCameraY(const cv::Vec3d &v, double yaw_rad)
{
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    return cv::Vec3d(
        c * v[0] + s * v[2],
        v[1],
        -s * v[0] + c * v[2]);
}

double Distance3d(const cv::Vec3d &a, const cv::Vec3d &b)
{
    const cv::Vec3d d = a - b;
    return std::sqrt(d.dot(d));
}

double ArmorYawRadFromPnp(const PnpResultMParam &pnp)
{
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << pnp.rvec[0], pnp.rvec[1], pnp.rvec[2]);
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);

    const double normal_x = rotation_matrix.at<double>(0, 0);
    const double normal_z = rotation_matrix.at<double>(2, 0);
    return std::atan2(normal_x, normal_z);
}

std::array<BodyCandidate, 4> GenerateBodyCandidates(const PnpResultMParam &pnp,
                                                    const RgoGeometry &geometry)
{
    const double armor_yaw_rad = ArmorYawRadFromPnp(pnp);
    const cv::Vec3d armor_pos = pnp.tvec_m;
    const std::array<ArmorSlot, 4> slots = {
        ArmorSlot::FRONT,
        ArmorSlot::LEFT,
        ArmorSlot::BACK,
        ArmorSlot::RIGHT};

    std::array<BodyCandidate, 4> candidates;
    for (std::size_t i = 0; i < slots.size(); ++i)
    {
        const ArmorSlot slot = slots[i];
        const double body_yaw_rad = armor_yaw_rad - SlotYawOffsetRad(slot);
        const cv::Vec3d offset_camera = RotateCameraY(SlotOffsetBody(slot, geometry), body_yaw_rad);

        candidates[i].slot = slot;
        candidates[i].yaw_rad = body_yaw_rad;
        candidates[i].center_m = armor_pos - offset_camera;
    }
    return candidates;
}

std::array<RgoArmor, 4> GenerateFourArmors(const cv::Vec3d &body_center_m,
                                           double body_yaw_rad,
                                           const RgoGeometry &geometry)
{
    const std::array<ArmorSlot, 4> slots = {
        ArmorSlot::FRONT,
        ArmorSlot::LEFT,
        ArmorSlot::BACK,
        ArmorSlot::RIGHT};

    std::array<RgoArmor, 4> armors;
    for (std::size_t i = 0; i < slots.size(); ++i)
    {
        const ArmorSlot slot = slots[i];
        armors[i].slot = slot;
        armors[i].center_m = body_center_m + RotateCameraY(SlotOffsetBody(slot, geometry), body_yaw_rad);
        armors[i].yaw_rad = body_yaw_rad + SlotYawOffsetRad(slot);
    }
    return armors;
}

BodyCandidate SelectBestCandidate(const std::array<BodyCandidate, 4> &candidates,
                                  bool has_state,
                                  const cv::Vec3d &last_center_m)
{
    if (!has_state)
    {
        BodyCandidate first = candidates[0];
        first.score = 0.0;
        return first;
    }

    BodyCandidate best = candidates[0];
    double best_score = Distance3d(candidates[0].center_m, last_center_m);
    for (std::size_t i = 1; i < candidates.size(); ++i)
    {
        const double score = Distance3d(candidates[i].center_m, last_center_m);
        if (score < best_score)
        {
            best = candidates[i];
            best_score = score;
        }
    }

    best.score = best_score;
    return best;
}

void DumpRgoForFoxglove(uint64_t frame_id,
                        int infer_id,
                        bool has_state,
                        const cv::Vec3d &body_center_m,
                        double body_yaw_rad,
                        ArmorSlot selected_slot,
                        double selected_score,
                        const std::array<RgoArmor, 4> &armors)
{
    if (!IsFoxgloveDebugEnabled())
    {
        return;
    }

    static const std::string kDir = "/tmp/rm_rerun";
    static const std::string kTmp = kDir + "/rgo_latest.tmp.json";
    static const std::string kOut = kDir + "/rgo_latest.json";
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
    fout << "  \"frame_id\": " << frame_id << ",\n";
    fout << "  \"infer_id\": " << infer_id << ",\n";
    fout << "  \"has_state\": " << (has_state ? "true" : "false") << ",\n";
    fout << "  \"selected_slot\": \"" << SlotName(selected_slot) << "\",\n";
    fout << "  \"selected_score\": " << selected_score << ",\n";
    fout << "  \"body_center_m\": [" << body_center_m[0] << ", " << body_center_m[1] << ", " << body_center_m[2] << "],\n";
    fout << "  \"body_yaw_rad\": " << body_yaw_rad << ",\n";
    fout << "  \"armors\": [\n";
    for (std::size_t i = 0; i < armors.size(); ++i)
    {
        const RgoArmor &armor = armors[i];
        fout << "    {\"slot\": \"" << SlotName(armor.slot) << "\", "
             << "\"center_m\": [" << armor.center_m[0] << ", " << armor.center_m[1] << ", " << armor.center_m[2] << "], "
             << "\"yaw_rad\": " << armor.yaw_rad << "}";
        fout << (i + 1 < armors.size() ? ",\n" : "\n");
    }
    fout << "  ]\n";
    fout << "}\n";
    fout.close();
    std::rename(kTmp.c_str(), kOut.c_str());
}

class RgoTrackerNode : public GNode
{
public:
    CStatus init() override
    {
        pnp_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(PnpResultMParam, PNP_TOPIC, 2);
        rgo_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(RgoOutputMParam, RGO_OUTPUT_TOPIC, 2);
        const int pool_size = GetAppConfig().rgo_message_pool_size > 0
                                  ? GetAppConfig().rgo_message_pool_size
                                  : 4;
        rgo_output_pool_.preallocate(static_cast<std::size_t>(pool_size));
        return CStatus();
    }

    CStatus run() override
    {
        const AppConfig &cfg = GetAppConfig();
        app::runtime::ApplyThreadAffinity("rgo", cfg.affinity_enable ? cfg.rgo_cpu : -1);
        while (!IsYoloStopRequested())
        {
            std::shared_ptr<PnpResultMParam> pnp = nullptr;
            const CStatus st = CGRAPH_SUB_MPARAM_WITH_TIMEOUT(PnpResultMParam, pnp_conn_id_, pnp, 1000);
            if (st.isErr() || !pnp)
            {
                continue;
            }

            std::shared_ptr<RgoOutputMParam> rgo = rgo_output_pool_.acquire();
            rgo->frame_id = pnp->frame_id;
            rgo->infer_id = pnp->infer_id;
            rgo->has_target = pnp->has_pose || has_state_;
            rgo->latency_ms = pnp->latency_ms;
            rgo->detect_latency_ms = pnp->detect_latency_ms;
            rgo->status = pnp->status;
            rgo->armor_type = pnp->armor_type;
            rgo->mean_reprojection_error_px = pnp->mean_reprojection_error_px;
            rgo->max_reprojection_error_px = pnp->max_reprojection_error_px;
            rgo->reprojection_ok = pnp->reprojection_ok;
            rgo->depth_ok = pnp->depth_ok;

            if (pnp->has_pose)
            {
                const std::array<BodyCandidate, 4> candidates = GenerateBodyCandidates(*pnp, geometry_);
                const BodyCandidate best = SelectBestCandidate(candidates, has_state_, body_center_m_);
                body_center_m_ = best.center_m;
                body_yaw_rad_ = best.yaw_rad;
                selected_slot_ = best.slot;
                selected_score_ = best.score;
                has_state_ = true;
            }

            rgo->xyz_m = has_state_ ? body_center_m_ : cv::Vec3d(0.0, 0.0, 0.0);
            rgo->yaw_deg = has_state_ ? body_yaw_rad_ * kRadToDeg : 0.0;
            rgo->bearing_yaw_deg =
                has_state_ ? std::atan2(body_center_m_[0], body_center_m_[2]) * kRadToDeg : 0.0;

            const std::array<RgoArmor, 4> armors =
                GenerateFourArmors(body_center_m_, body_yaw_rad_, geometry_);
            DumpRgoForFoxglove(pnp->frame_id,
                                pnp->infer_id,
                                has_state_,
                                body_center_m_,
                                body_yaw_rad_,
                                selected_slot_,
                                selected_score_,
                                armors);

            const CStatus pub_st = CGRAPH_PUB_MPARAM(RgoOutputMParam, RGO_OUTPUT_TOPIC, rgo, GMessagePushStrategy::REPLACE);
            if (pub_st.isErr())
            {
                spdlog::warn("[RGO] publish output failed frame={} infer={} error={}",
                             pnp->frame_id,
                             pnp->infer_id,
                             pub_st.getInfo());
            }
        }
        return CStatus();
    }

private:
    int pnp_conn_id_ = -1;
    int rgo_conn_id_ = -1;
    SharedParamPool<RgoOutputMParam> rgo_output_pool_;

    bool has_state_ = false;
    cv::Vec3d body_center_m_ = cv::Vec3d(0.0, 0.0, 0.0);
    double body_yaw_rad_ = 0.0;
    ArmorSlot selected_slot_ = ArmorSlot::FRONT;
    double selected_score_ = 0.0;
    RgoGeometry geometry_;
};

}  // namespace

void RegisterRgoPipelineElements(CGraph::GPipeline* const &pipeline,
                                 CGraph::GElementPtr *rgo_ref,
                                 const CGraph::GElementPtrSet &depends)
{
    GElementPtr rgo = nullptr;
    CStatus st = pipeline->registerGElement<RgoTrackerNode>(&rgo, depends, "RGO 跟踪\nrgo_tracker");
    if (st.isErr())
    {
        spdlog::error("register rgo tracker failed: {}", st.getInfo());
    }
    if (rgo_ref)
    {
        *rgo_ref = rgo;
    }
}
