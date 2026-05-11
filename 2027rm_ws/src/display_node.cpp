#include "display_node.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "yolo_common.hpp"

CStatus DisplayNode::init()
{
    result_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(ResultMParam, RESULT_TOPIC, 2);
    pnp_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(PnpResultMParam, PNP_TOPIC, 2);
    fps_start_ = std::chrono::steady_clock::now();
    log_start_ = fps_start_;
    count_ = 0;
    log_count_ = 0;
    last_counted_frame_id_ = 0;
    det_sum_ = 0;
    fps_ = 0.0;
    latency_sum_ms_ = 0.0;
    latency_max_ms_ = 0.0;
    detect_latency_sum_ms_ = 0.0;
    detect_latency_max_ms_ = 0.0;
    dropped_result_count_ = 0;
    result_timeout_count_ = 0;
    return CStatus();
}

CStatus DisplayNode::run()
{
    const bool enable_vis = IsFoxgloveDebugEnabled();
    while (!IsYoloStopRequested())
    {
        std::shared_ptr<ResultMParam> r = nullptr;
        CStatus st = CGRAPH_SUB_MPARAM_WITH_TIMEOUT(ResultMParam, result_conn_id_, r, 1000);
        if (st.isErr())
        {
            if (IsYoloStopRequested())
            {
                break;
            }
            ++result_timeout_count_;
            const auto now = std::chrono::steady_clock::now();
            const auto log_dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - log_start_).count();
            if (IsE2eLogEnabled() && log_dt_ms >= 1000)
            {
                spdlog::info("[E2E] fps=0 latency_ms_avg=0 latency_ms_max=0 detect_latency_ms_avg=0 detect_latency_ms_max=0 det=0 dropped_stale={} result_timeout={}",
                             dropped_result_count_,
                             result_timeout_count_);
                log_count_ = 0;
                det_sum_ = 0;
                latency_sum_ms_ = 0.0;
                latency_max_ms_ = 0.0;
                detect_latency_sum_ms_ = 0.0;
                detect_latency_max_ms_ = 0.0;
                dropped_result_count_ = 0;
                result_timeout_count_ = 0;
                log_start_ = now;
            }
            continue;
        }
        if (!r)
        {
            continue;
        }
        if (r->frame_id <= last_counted_frame_id_)
        {
            ++dropped_result_count_;
            spdlog::debug("[DISPLAY] show out-of-order result frame={} latest={}",
                          r->frame_id,
                          last_counted_frame_id_);
        }
        else
        {
            last_counted_frame_id_ = r->frame_id;
        }
        ++count_;
        ++log_count_;
        det_sum_ += r->det_count;
        auto now = std::chrono::steady_clock::now();
        const double full_latency_ms = ElapsedMsSince(r->pipeline_start_tp);
        latency_sum_ms_ += full_latency_ms;
        latency_max_ms_ = std::max(latency_max_ms_, full_latency_ms);
        detect_latency_sum_ms_ += r->detect_latency_ms;
        detect_latency_max_ms_ = std::max(detect_latency_max_ms_, r->detect_latency_ms);
        auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start_).count();
        if (dt_ms >= 1000)
        {
            fps_ = count_ * 1000.0 / static_cast<double>(dt_ms);
            count_ = 0;
            fps_start_ = now;
        }

        auto log_dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - log_start_).count();
        if (IsE2eLogEnabled() && log_dt_ms >= 1000)
        {
            double real_fps = log_count_ * 1000.0 / static_cast<double>(log_dt_ms);
            const double latency_avg_ms = (log_count_ > 0) ? (latency_sum_ms_ / static_cast<double>(log_count_)) : 0.0;
            const double detect_latency_avg_ms = (log_count_ > 0) ? (detect_latency_sum_ms_ / static_cast<double>(log_count_)) : 0.0;
            const double det_avg = (log_count_ > 0) ? (static_cast<double>(det_sum_) / static_cast<double>(log_count_)) : 0.0;
            spdlog::info("[E2E] fps={} latency_ms_avg={} latency_ms_max={} detect_latency_ms_avg={} detect_latency_ms_max={} det={} dropped_stale={}",
                         static_cast<int>(real_fps + 0.5),
                         static_cast<int>(latency_avg_ms + 0.5),
                         static_cast<int>(latency_max_ms_ + 0.5),
                         static_cast<int>(detect_latency_avg_ms + 0.5),
                         static_cast<int>(detect_latency_max_ms_ + 0.5),
                         cv::format("%.1f", det_avg),
                         dropped_result_count_);
            log_count_ = 0;
            det_sum_ = 0;
            latency_sum_ms_ = 0.0;
            latency_max_ms_ = 0.0;
            detect_latency_sum_ms_ = 0.0;
            detect_latency_max_ms_ = 0.0;
            dropped_result_count_ = 0;
            result_timeout_count_ = 0;
            log_start_ = now;
        }

        if (!enable_vis || r->vis.empty())
        {
            continue;
        }

        std::shared_ptr<PnpResultMParam> matched_pnp = DrainPnpResult(r->frame_id);
        if (!matched_pnp && latest_pnp_ && latest_pnp_->frame_id <= r->frame_id)
        {
            matched_pnp = latest_pnp_;
        }

        cv::Mat &show = r->vis;
        std::string t1 = "frame=" + std::to_string(r->frame_id) + " infer=" + std::to_string(r->infer_id) + " det=" + std::to_string(r->det_count);
        std::string t2 = "fps=" + std::to_string(static_cast<int>(fps_ + 0.5));
        std::string t3 = "latency=" + std::to_string(static_cast<int>(full_latency_ms + 0.5)) +
                         "ms detect=" + std::to_string(static_cast<int>(r->detect_latency_ms + 0.5)) + "ms";
        std::string t4 = "corners=none";
        std::string t5 = "pnp=none";
        std::string t6;
        if (r->has_corners)
        {
            t4 = "corners=[(" +
                 std::to_string(static_cast<int>(std::round(r->corners[0].x))) + "," +
                 std::to_string(static_cast<int>(std::round(r->corners[0].y))) + "),(" +
                 std::to_string(static_cast<int>(std::round(r->corners[1].x))) + "," +
                 std::to_string(static_cast<int>(std::round(r->corners[1].y))) + "),(" +
                 std::to_string(static_cast<int>(std::round(r->corners[2].x))) + "," +
                 std::to_string(static_cast<int>(std::round(r->corners[2].y))) + "),(" +
                 std::to_string(static_cast<int>(std::round(r->corners[3].x))) + "," +
                 std::to_string(static_cast<int>(std::round(r->corners[3].y))) + ")]";
            for (const auto &corner_pt : r->corners)
            {
                cv::circle(show, corner_pt, 5, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
            }
        }
        if (matched_pnp)
        {
            const auto &p = matched_pnp;
            const int64_t pnp_lag = static_cast<int64_t>(r->frame_id) - static_cast<int64_t>(p->frame_id);
            t5 = "pnp=" + (p->has_pose ? p->armor_type : p->status) +
                 " frame=" + std::to_string(p->frame_id) +
                 " lag=" + std::to_string(pnp_lag);
            if (p->has_pose)
            {
                if (!p->status.empty())
                {
                    t5 += " " + p->status;
                }
                t5 += " center=(" + std::to_string(static_cast<int>(std::round(p->center_px.x))) +
                      "," + std::to_string(static_cast<int>(std::round(p->center_px.y))) + ")" +
                      " z=" + cv::format("%.2f", p->tvec_m[2]) + "m" +
                      " latency=" + std::to_string(static_cast<int>(p->latency_ms + 0.5)) + "ms";
                t6 = "xyz=(" + cv::format("%.2f", p->tvec_m[0]) + "," +
                     cv::format("%.2f", p->tvec_m[1]) + "," +
                     cv::format("%.2f", p->tvec_m[2]) + ") ypr=(" +
                     cv::format("%.1f", p->ypr_deg[0]) + "," +
                     cv::format("%.1f", p->ypr_deg[1]) + "," +
                     cv::format("%.1f", p->ypr_deg[2]) + ")";
                cv::circle(show, p->center_px, 7, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
                cv::drawMarker(show, p->center_px, cv::Scalar(255, 255, 0), cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
            }
        }
        cv::putText(show, t1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::putText(show, t2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(show, t3, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
        cv::putText(show, t4, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
        cv::putText(show, t5, cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        if (!t6.empty())
        {
            cv::putText(show, t6, cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 220, 120), 1, cv::LINE_AA);
        }
        DumpFrameForFoxglove(show);
    }
    return CStatus();
}

CStatus DisplayNode::destroy()
{
    return CStatus();
}

std::shared_ptr<PnpResultMParam> DisplayNode::DrainPnpResult(uint64_t frame_id)
{
    while (true)
    {
        std::shared_ptr<PnpResultMParam> pnp = nullptr;
        const CStatus pnp_st = CGRAPH_SUB_MPARAM_WITH_TIMEOUT(PnpResultMParam, pnp_conn_id_, pnp, 0);
        if (pnp_st.isErr())
        {
            break;
        }
        if (!pnp)
        {
            break;
        }
        if (pnp->frame_id < frame_id)
        {
            latest_pnp_ = pnp;
            continue;
        }
        if (pnp->frame_id == frame_id)
        {
            latest_pnp_ = pnp;
            return pnp;
        }
        latest_pnp_ = pnp;
        break;
    }
    return nullptr;
}
