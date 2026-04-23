#include <spdlog/spdlog.h>

#include "logging.hpp"
#include <CGraph.h>

#include "thread_startup.hpp"
#include "yolo_app.hpp"

int main() {
    app::logging::InitAsyncLogging();

    const AppConfig &cfg = GetAppConfig();

    spdlog::info("model={} device={} classes={} conf={} nms={} threads={} workers={} target_color={}",
                 cfg.model_path,
                 cfg.device,
                 cfg.num_classes,
                 cfg.conf_thres,
                 cfg.nms_thres,
                 cfg.thread_num,
                 cfg.infer_workers,
                 cfg.target_color);

    InitYoloMessageTopics();
    int ret = app::RunThreadedPipeline(cfg.thread_num, RegisterYoloPipelineElements);
    ShutdownYoloApp();
    return ret;
}
