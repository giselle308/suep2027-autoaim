#pragma once

#include <string>

#include <CGraph.h>

struct AppConfig {
    std::string model_path;
    std::string device;
    int num_classes;
    float conf_thres;
    float nms_thres;
    int thread_num;
    int infer_workers;
    std::string target_color;
};

const AppConfig& GetAppConfig();
CStatus RegisterYoloPipelineElements(CGraph::GPipeline* const &pipeline);
void InitYoloMessageTopics();
void ShutdownYoloApp();
