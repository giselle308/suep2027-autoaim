#include <spdlog/spdlog.h>

#include "logging.hpp"
#include <CGraph.h>

#include "thread_startup.hpp"
#include "yolo_app.hpp"

int main() {
    app::logging::InitAsyncLogging();

    const AppConfig &cfg = GetAppConfig();

    spdlog::info("model={} device={} classes={} conf={} nms={} fast_nms_topk={} threads={} workers={} target_color={} pnp_enable={} pnp_small_width_m={} pnp_big_width_m={} pnp_armor_length_m={} pnp_lightbar_length_m={} pnp_rigid_constraint={} pnp_reproj_mean_px={} pnp_reproj_corner_px={} pnp_depth_range_m=[{},{}]",
                 cfg.model_path,
                 cfg.device,
                 cfg.num_classes,
                 cfg.conf_thres,
                 cfg.nms_thres,
                 cfg.fast_nms_topk,
                 cfg.thread_num,
                 cfg.infer_workers,
                 cfg.target_color,
                 cfg.pnp_enable,
                 cfg.pnp_small_armor_width_m,
                 cfg.pnp_big_armor_width_m,
                 cfg.pnp_armor_length_m,
                 cfg.pnp_lightbar_length_m,
                 cfg.pnp_rigid_constraint_enable,
                 cfg.pnp_max_mean_reprojection_error_px,
                 cfg.pnp_max_corner_reprojection_error_px,
                 cfg.pnp_min_depth_m,
                 cfg.pnp_max_depth_m);

    InitYoloMessageTopics();
    int ret = app::RunThreadedPipeline(cfg.thread_num, RegisterYoloPipelineElements);
    ShutdownYoloApp();
    return ret;
}
