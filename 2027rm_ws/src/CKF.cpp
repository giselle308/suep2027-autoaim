#include <chrono>
#include <cmath>
#include <memory>

#include <CGraph.h>
#include <spdlog/spdlog.h>

#include "yolo_app.hpp"
#include "yolo_common.hpp"

using namespace CGraph;

namespace {

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

class CkfInputNode : public GNode
{
public:
    CStatus init() override
    {
        pnp_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(PnpResultMParam, PNP_TOPIC, 2);
        stat_start_ = std::chrono::steady_clock::now();
        recv_count_ = 0;
        output_count_ = 0;
        no_pose_count_ = 0;
        return CStatus();
    }

    CStatus run() override
    {
        while (!IsYoloStopRequested())
        {
            std::shared_ptr<PnpResultMParam> pnp = nullptr;
            const CStatus st = CGRAPH_SUB_MPARAM_WITH_TIMEOUT(PnpResultMParam, pnp_conn_id_, pnp, 1000);
            if (st.isErr())
            {
                LogStatsIfDue();
                continue;
            }
            if (!pnp)
            {
                LogStatsIfDue();
                continue;
            }

            ++recv_count_;
            const auto ckf_start = std::chrono::steady_clock::now();
            CkfInputMParam ckf;
            ckf.frame_id = pnp->frame_id;
            ckf.infer_id = pnp->infer_id;
            ckf.has_target = pnp->has_pose;
            ckf.detect_latency_ms = pnp->detect_latency_ms;
            ckf.camera_grab_ms = pnp->camera_grab_ms;
            ckf.pixel_convert_ms = pnp->pixel_convert_ms;
            ckf.preprocess_ms = pnp->preprocess_ms;
            ckf.infer_async_ms = pnp->infer_async_ms;
            ckf.postprocess_ms = pnp->postprocess_ms;
            ckf.pnp_solve_ms = pnp->pnp_solve_ms;
            ckf.pnp_total_ms = pnp->pnp_total_ms;
            ckf.status = pnp->status;
            ckf.armor_type = pnp->armor_type;

            if (pnp->has_pose)
            {
                ckf.xyz_m = pnp->tvec_m;
                ckf.yaw_deg = pnp->ypr_deg[0];
                ckf.bearing_yaw_deg = std::atan2(pnp->tvec_m[0], pnp->tvec_m[2]) * kRadToDeg;
            }
            else
            {
                ++no_pose_count_;
                ckf.xyz_m = cv::Vec3d(0.0, 0.0, 0.0);
                ckf.yaw_deg = 0.0;
                ckf.bearing_yaw_deg = 0.0;
            }

            ckf.ckf_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ckf_start).count();
            ckf.latency_ms = ElapsedMsSince(pnp->capture_tp);

            ++output_count_;
            total_latency_stat_.add(ckf.latency_ms);
            detect_latency_stat_.add(pnp->detect_latency_ms);
            pixel_convert_stat_.add(pnp->pixel_convert_ms);
            preprocess_stat_.add(pnp->preprocess_ms);
            infer_async_stat_.add(pnp->infer_async_ms);
            postprocess_stat_.add(pnp->postprocess_ms);
            pnp_solve_stat_.add(pnp->pnp_solve_ms);
            pnp_total_stat_.add(pnp->pnp_total_ms);
            ckf_stat_.add(ckf.ckf_ms);
            LogStatsIfDue();
        }
        return CStatus();
    }

private:
    void LogStatsIfDue()
    {
        const auto now = std::chrono::steady_clock::now();
        const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - stat_start_).count();
        if (dt_ms < 1000)
        {
            return;
        }
        const double fps = output_count_ * 1000.0 / static_cast<double>(dt_ms);
        spdlog::info("[LINK] fps={:.1f} recv={} output={} no_pose={} total_no_camera_avg={:.2f}ms total_no_camera_max={:.2f}ms detect_avg={:.2f}ms detect_max={:.2f}ms pixel_convert_avg={:.2f}ms pixel_convert_max={:.2f}ms preprocess_avg={:.2f}ms preprocess_max={:.2f}ms infer_async_avg={:.2f}ms infer_async_max={:.2f}ms postprocess_avg={:.2f}ms postprocess_max={:.2f}ms pnp_solve_avg={:.2f}ms pnp_solve_max={:.2f}ms pnp_total_avg={:.2f}ms pnp_total_max={:.2f}ms ckf_avg={:.2f}ms ckf_max={:.2f}ms",
                     fps,
                     recv_count_,
                     output_count_,
                     no_pose_count_,
                     total_latency_stat_.avg(),
                     total_latency_stat_.max_ms,
                     detect_latency_stat_.avg(),
                     detect_latency_stat_.max_ms,
                     pixel_convert_stat_.avg(),
                     pixel_convert_stat_.max_ms,
                     preprocess_stat_.avg(),
                     preprocess_stat_.max_ms,
                     infer_async_stat_.avg(),
                     infer_async_stat_.max_ms,
                     postprocess_stat_.avg(),
                     postprocess_stat_.max_ms,
                     pnp_solve_stat_.avg(),
                     pnp_solve_stat_.max_ms,
                     pnp_total_stat_.avg(),
                     pnp_total_stat_.max_ms,
                     ckf_stat_.avg(),
                     ckf_stat_.max_ms);
        recv_count_ = 0;
        output_count_ = 0;
        no_pose_count_ = 0;
        ResetLatencyStats();
        stat_start_ = now;
    }

    struct LatencyStat
    {
        int count = 0;
        double sum_ms = 0.0;
        double max_ms = 0.0;

        void add(double ms)
        {
            ++count;
            sum_ms += ms;
            if (ms > max_ms)
            {
                max_ms = ms;
            }
        }

        double avg() const
        {
            return count > 0 ? sum_ms / static_cast<double>(count) : 0.0;
        }

        void reset()
        {
            count = 0;
            sum_ms = 0.0;
            max_ms = 0.0;
        }
    };

    void ResetLatencyStats()
    {
        total_latency_stat_.reset();
        detect_latency_stat_.reset();
        pixel_convert_stat_.reset();
        preprocess_stat_.reset();
        infer_async_stat_.reset();
        postprocess_stat_.reset();
        pnp_solve_stat_.reset();
        pnp_total_stat_.reset();
        ckf_stat_.reset();
    }

    int pnp_conn_id_ = -1;
    std::chrono::steady_clock::time_point stat_start_;
    int recv_count_ = 0;
    int output_count_ = 0;
    int no_pose_count_ = 0;
    LatencyStat total_latency_stat_;
    LatencyStat detect_latency_stat_;
    LatencyStat pixel_convert_stat_;
    LatencyStat preprocess_stat_;
    LatencyStat infer_async_stat_;
    LatencyStat postprocess_stat_;
    LatencyStat pnp_solve_stat_;
    LatencyStat pnp_total_stat_;
    LatencyStat ckf_stat_;
};

}  // namespace

void RegisterCkfPipelineElements(CGraph::GPipeline* const &pipeline,
                                 CGraph::GElementPtr *ckf_ref,
                                 const CGraph::GElementPtrSet &depends)
{
    GElementPtr ckf = nullptr;
    CStatus st = pipeline->registerGElement<CkfInputNode>(&ckf, depends, "CKF 输入\nckf_input");
    if (st.isErr())
    {
        spdlog::error("register ckf input failed: {}", st.getInfo());
    }
    if (ckf_ref)
    {
        *ckf_ref = ckf;
    }
}
