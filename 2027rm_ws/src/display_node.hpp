#pragma once

#include <chrono>
#include <memory>

#include <CGraph.h>

#include "yolo_app.hpp"

class DisplayNode : public CGraph::GNode
{
public:
    CStatus init() override;
    CStatus run() override;
    CStatus destroy() override;

private:
    std::shared_ptr<PnpResultMParam> DrainPnpResult(uint64_t frame_id);

    int result_conn_id_ = -1;
    int pnp_conn_id_ = -1;
    std::shared_ptr<PnpResultMParam> latest_pnp_ = nullptr;
    std::chrono::steady_clock::time_point fps_start_;
    std::chrono::steady_clock::time_point log_start_;
    uint64_t last_counted_frame_id_ = 0;
    int count_ = 0;
    int log_count_ = 0;
    int det_sum_ = 0;
    int dropped_result_count_ = 0;
    double fps_ = 0.0;
    double latency_sum_ms_ = 0.0;
    double latency_max_ms_ = 0.0;
};
