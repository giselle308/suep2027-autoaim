#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <CGraph.h>

#include "camera_node.hpp"
#include "display_node.hpp"
#include "yolo_app.hpp"
#include "yolo_common.hpp"
#include "yolo_openvino.hpp"

using namespace CGraph;

class CameraPubNode : public GNode
{
private:
    HikCameraNode camera_;
    uint64_t frame_id_ = 0;

public:
    CStatus init() override
    {
        std::string err;
        if (!camera_.init(&err))
        {
            return CStatus(err);
        }
        frame_id_ = 0;
        return CStatus();
    }

    CStatus run() override
    {
        while (!IsYoloStopRequested())
        {
            cv::Mat frame;
            std::chrono::steady_clock::time_point capture_tp;
            std::string err;
            if (!camera_.grab(frame, &capture_tp, &err))
            {
                return CStatus(err);
            }
            std::shared_ptr<FrameMParam> msg(new FrameMParam());
            msg->frame = std::move(frame);
            msg->frame_id = ++frame_id_;
            msg->capture_tp = capture_tp;
            CStatus st = CGRAPH_SEND_MPARAM(FrameMParam, FRAME_TOPIC, msg, GMessagePushStrategy::REPLACE);
            if (st.isErr())
            {
                return st;
            }
        }
        return CStatus();
    }

    CStatus destroy() override
    {
        camera_.shutdown();
        return CStatus();
    }
};

class YoloInferNode : public GNode
{
private:
    YoloOpenvino yolo_;
    YoloResultFilter result_filter_;

    CStatus publishResults(const std::vector<std::shared_ptr<ResultMParam>> &results)
    {
        for (const auto &out : results)
        {
            if (!out)
            {
                continue;
            }

            result_filter_.apply(*out, GetAppConfig());
            if (out->has_corners && !out->vis.empty())
            {
                DrawArmorFrame(out->vis, out->corners, cv::Scalar(0, 255, 255));
            }

            CStatus st = CGRAPH_PUB_MPARAM(ResultMParam, RESULT_TOPIC, out, GMessagePushStrategy::REPLACE);
            if (st.isErr())
            {
                return st;
            }
        }
        return CStatus();
    }

public:
    CStatus init() override
    {
        std::string err;
        if (!yolo_.init(GetAppConfig(), &err))
        {
            return CStatus(err);
        }
        return CStatus();
    }

    CStatus run() override
    {
        while (!IsYoloStopRequested())
        {
            std::vector<std::shared_ptr<ResultMParam>> ready_results;
            std::string err;
            if (!yolo_.collectCompleted(ready_results, &err))
            {
                return CStatus(err);
            }
            CStatus st = publishResults(ready_results);
            if (st.isErr())
            {
                return st;
            }

            if (!yolo_.hasIdleRequest())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            std::shared_ptr<FrameMParam> in = nullptr;
            st = CGRAPH_RECV_MPARAM_WITH_TIMEOUT(FrameMParam, FRAME_TOPIC, in, 1);
            if (st.isErr())
            {
                continue;
            }
            if (!in || in->frame.empty())
            {
                continue;
            }
            if (!yolo_.submit(*in, &err))
            {
                return CStatus(err);
            }
        }

        std::vector<std::shared_ptr<ResultMParam>> tail_results;
        std::string err;
        if (!yolo_.waitAll(tail_results, &err))
        {
            return CStatus(err);
        }
        CStatus st = publishResults(tail_results);
        if (st.isErr())
        {
            return st;
        }
        return CStatus();
    }
};

CStatus RegisterYoloPipelineElements(CGraph::GPipeline* const &pipeline)
{
    GElementPtr cam = nullptr;
    GElementPtr infer = nullptr;
    GElementPtr pnp = nullptr;
    GElementPtr display = nullptr;
    CStatus st;
    const bool dump_only = std::getenv("CGRAPH_DUMP_ONLY") != nullptr;
    const std::string camera_name = dump_only ? "相机发布\ncamera_pub" : "camera_pub";
    const std::string infer_name = dump_only ? "OpenVINO 异步推理\nasync_infer" : "async_infer";
    const std::string display_name = dump_only ? "显示/调试\ndisplay" : "display";

    st += pipeline->registerGElement<CameraPubNode>(&cam, {}, camera_name);
    const GElementPtrSet infer_depends = dump_only ? GElementPtrSet{cam} : GElementPtrSet{};
    st += pipeline->registerGElement<YoloInferNode>(&infer, infer_depends, infer_name);
    GElementPtrSet result_depends = {infer};
    RegisterPnpPipelineElements(pipeline, &pnp, dump_only ? result_depends : GElementPtrSet{});
    if (dump_only && pnp)
    {
        result_depends.insert(pnp);
    }
    st += pipeline->registerGElement<DisplayNode>(&display, dump_only ? result_depends : GElementPtrSet{}, display_name);
    return st;
}

void InitYoloMessageTopics()
{
    CGRAPH_CREATE_MESSAGE_TOPIC(FrameMParam, FRAME_TOPIC, 2);
}

void ShutdownYoloApp()
{
    RequestYoloStop();
    ClearYoloMessages();
}
