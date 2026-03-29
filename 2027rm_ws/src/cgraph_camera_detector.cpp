#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "CGraph.h"
#include <opencv2/opencv.hpp>

#include "camera_node.hpp"
#include "detector_node.hpp"

using namespace CGraph;

static const char *FRAME_PARAM_KEY = "frame-param-key";

struct AppConfig
{
    float depth_cm = 100.0f;
#ifdef CAMERA_INFO_PATH
    std::string camera_info = CAMERA_INFO_PATH;
#else
    std::string camera_info = "../config/camera_info.yaml";
#endif
    float expected_len_cm = 6.0f;
    float expected_gap_cm = 14.0f;
} g_cfg;

static bool FileExists(const std::string &path)
{
    std::ifstream fin(path);
    return fin.good();
}

static std::string ResolveCameraInfoPath(const std::string &inputPath)
{
    const std::string requested = inputPath.empty() ? g_cfg.camera_info : inputPath;
    const std::string candidates[] = {
        requested,
#ifdef CAMERA_INFO_PATH
        CAMERA_INFO_PATH,
#endif
        "../config/camera_info.yaml",
        "2027rm_ws/config/camera_info.yaml",
        "config/camera_info.yaml"};

    for (const auto &p : candidates)
    {
        if (!p.empty() && FileExists(p))
        {
            return p;
        }
    }
    return requested;
}

struct FrameGParam : public GParam
{
    cv::Mat frame;
    int frame_id = 0;
};

class CameraNode : public GNode
{
public:
    CStatus init() override
    {
        std::string error;
        if (!camera_.init(&error))
        {
            return CStatus(error);
        }
        return CStatus();
    }

    CStatus run() override
    {
        cv::Mat frame;
        std::string error;
        if (!camera_.grab(frame, &error))
        {
            return CStatus(error);
        }

        auto param = CGRAPH_GET_GPARAM_WITH_NO_EMPTY(FrameGParam, FRAME_PARAM_KEY);
        param->frame = frame;
        param->frame_id++;
        return CStatus();
    }

    CStatus destroy() override
    {
        camera_.shutdown();
        return CStatus();
    }

private:
    HikCameraNode camera_;
};

class DetectorNode : public GNode
{
public:
    CStatus init() override
    {
        std::string error;
        if (!detector_.init(g_cfg.depth_cm,
                            g_cfg.camera_info,
                            g_cfg.expected_len_cm,
                            g_cfg.expected_gap_cm,
                            &error))
        {
            return CStatus(error);
        }
        fps_start_ = std::chrono::steady_clock::now();
        frame_count_ = 0;
        fps_ = 0.0;
        return CStatus();
    }

    CStatus run() override
    {
        auto param = CGRAPH_GET_GPARAM_WITH_NO_EMPTY(FrameGParam, FRAME_PARAM_KEY);
        if (param->frame.empty())
        {
            return CStatus("empty frame in detector");
        }

        cv::Mat vis;
        bool found = false;
        std::string error;
        if (!detector_.detect(param->frame, vis, &found, &error))
        {
            return CStatus(error);
        }

        ++frame_count_;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start_).count();
        if (elapsed_ms >= 1000)
        {
            fps_ = frame_count_ * 1000.0 / static_cast<double>(elapsed_ms);
            frame_count_ = 0;
            fps_start_ = now;
            std::cout << "CGraph FPS: " << static_cast<int>(fps_ + 0.5) << std::endl;
        }

        const std::string info = std::string("frame=") + std::to_string(param->frame_id) +
                                 " depth=" + std::to_string(static_cast<int>(g_cfg.depth_cm + 0.5f)) + "cm";
        cv::putText(vis, info, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

        const std::string fps_text = std::string("FPS=") + std::to_string(static_cast<int>(fps_ + 0.5));
        cv::putText(vis, fps_text, cv::Point(20, 105), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        cv::imshow("CGraph Camera->Detector", vis);
        const int key = cv::waitKey(1);
        if (key == 27)
        {
            return CStatus("user stopped by ESC");
        }
        return CStatus();
    }

private:
    RedLightBarDetector detector_;
    std::chrono::steady_clock::time_point fps_start_;
    int frame_count_ = 0;
    double fps_ = 0.0;
};

static void ParseArgs(int argc, char **argv)
{
    if (argc >= 2)
        g_cfg.depth_cm = std::stof(argv[1]);
    if (argc >= 3)
    {
        g_cfg.camera_info = ResolveCameraInfoPath(argv[2]);
    }
    else
    {
        g_cfg.camera_info = ResolveCameraInfoPath(g_cfg.camera_info);
    }
    if (argc >= 4)
        g_cfg.expected_len_cm = std::stof(argv[3]);
    if (argc >= 5)
        g_cfg.expected_gap_cm = std::stof(argv[4]);
}

int main(int argc, char **argv)
{
    ParseArgs(argc, argv);

    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr camera = nullptr;
    GElementPtr detector = nullptr;

    CStatus status;
    status += pipeline->registerGElement<CameraNode>(&camera, {}, "camera_node");
    status += pipeline->registerGElement<DetectorNode>(&detector, {camera}, "detector_node");
    status += pipeline->createGParam<FrameGParam>(FRAME_PARAM_KEY);

    if (status.isErr())
    {
        std::cerr << "pipeline build failed: " << status.getInfo() << std::endl;
        GPipelineFactory::clear();
        return -1;
    }

    {
        std::ofstream dot_file("2027rm_ws/output/cgraph_pipeline.dot");
        if (dot_file.good())
        {
            CStatus dump_status = pipeline->dump(dot_file);
            if (dump_status.isErr())
            {
                std::cerr << "pipeline dump file failed: " << dump_status.getInfo() << std::endl;
            }
            else
            {
                std::cout << "pipeline graph exported to 2027rm_ws/output/cgraph_pipeline.dot" << std::endl;
            }
        }
    }

    std::cout << "Start CGraph camera-detector pipeline" << std::endl;
    std::cout << "depth_cm=" << g_cfg.depth_cm
              << " camera_info=" << g_cfg.camera_info
              << " len_cm=" << g_cfg.expected_len_cm
              << " gap_cm=" << g_cfg.expected_gap_cm << std::endl;

    status = pipeline->process(1000000);
    if (status.isErr())
    {
        std::cout << "pipeline exit: " << status.getInfo() << std::endl;
    }

    GPipelineFactory::clear();
    cv::destroyAllWindows();
    return 0;
}
