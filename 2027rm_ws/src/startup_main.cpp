#include <iostream>

#include <CGraph.h>

#include "thread_startup.hpp"
#include "yolo_app.hpp"

int main() {
    const AppConfig &cfg = GetAppConfig();

    std::cout << "model=" << cfg.model_path << " device=" << cfg.device << " classes=" << cfg.num_classes
              << " conf=" << cfg.conf_thres << " nms=" << cfg.nms_thres << " threads=" << cfg.thread_num
              << " workers=" << cfg.infer_workers << " target_color=" << cfg.target_color << std::endl;

    InitYoloMessageTopics();
    int ret = app::RunThreadedPipeline(cfg.thread_num, RegisterYoloPipelineElements);
    ShutdownYoloApp();
    return ret;
}
