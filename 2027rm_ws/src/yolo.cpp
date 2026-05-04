#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <CGraph.h>

#include "camera_node.hpp"
#include "display_node.hpp"
#include "message_pool.hpp"
#include "profiling.hpp"
#include "yolo_app.hpp"
#include "yolo_common.hpp"
#include "yolo_openvino.hpp"

using namespace CGraph;

namespace {

bool IsReusableMatBuffer(const cv::Mat &mat)
{
    return mat.empty() || mat.u == nullptr || mat.u->refcount == 1;
}

}  // namespace

class CameraPubNode : public GNode
{
private:
    HikCameraNode camera_;
    uint64_t frame_id_ = 0;
    std::vector<cv::Mat> frame_pool_;
    std::size_t next_frame_slot_ = 0;
    SharedParamPool<FrameMParam> frame_msg_pool_;

public:
    CStatus init() override
    {
        std::string err;
        if (!camera_.init(&err))
        {
            return CStatus(err);
        }
        frame_id_ = 0;
        frame_pool_.resize(static_cast<std::size_t>(std::max(4, GetAppConfig().infer_workers + 3)));
        next_frame_slot_ = 0;
        return CStatus();
    }

    CStatus run() override
    {
        while (!IsYoloStopRequested())
        {
            cv::Mat &frame = acquireFrameBuffer();
            std::chrono::steady_clock::time_point capture_tp;
            std::string err;
            if (!camera_.grab(frame, &capture_tp, &err))
            {
                return CStatus(err);
            }
            std::shared_ptr<FrameMParam> msg = frame_msg_pool_.acquire();
            msg->frame = frame;
            msg->frame_id = ++frame_id_;
            msg->capture_tp = capture_tp;
            CStatus st = CGRAPH_SEND_MPARAM(FrameMParam, FRAME_TOPIC, msg, GMessagePushStrategy::REPLACE);
            if (st.isErr())
            {
                return st;
            }
            app::profiling::LogIfDue();
        }
        return CStatus();
    }

    CStatus destroy() override
    {
        camera_.shutdown();
        frame_pool_.clear();
        return CStatus();
    }

private:
    cv::Mat &acquireFrameBuffer()
    {
        for (std::size_t i = 0; i < frame_pool_.size(); ++i)
        {
            const std::size_t idx = (next_frame_slot_ + i) % frame_pool_.size();
            if (IsReusableMatBuffer(frame_pool_[idx]))
            {
                next_frame_slot_ = (idx + 1) % frame_pool_.size();
                return frame_pool_[idx];
            }
        }

        frame_pool_.emplace_back();
        next_frame_slot_ = 0;
        return frame_pool_.back();
    }
};

class YoloInferNode : public GNode
{
private:
    YoloOpenvino yolo_;
    YoloResultFilter result_filter_;
    std::vector<std::shared_ptr<ResultMParam>> ready_results_;
    std::vector<std::shared_ptr<ResultMParam>> tail_results_;

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
        ready_results_.reserve(static_cast<std::size_t>(std::max(1, GetAppConfig().infer_workers)));
        tail_results_.reserve(static_cast<std::size_t>(std::max(1, GetAppConfig().infer_workers)));
        return CStatus();
    }

    CStatus run() override
    {
        while (!IsYoloStopRequested())
        {
            ready_results_.clear();
            std::string err;
            if (!yolo_.collectCompleted(ready_results_, &err))
            {
                return CStatus(err);
            }
            CStatus st = publishResults(ready_results_);
            if (st.isErr())
            {
                return st;
            }

            if (!yolo_.hasIdleRequest())
            {
                yolo_.waitForCompletionSignal(std::chrono::milliseconds(1));
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
            app::profiling::LogIfDue();
        }

        tail_results_.clear();
        std::string err;
        if (!yolo_.waitAll(tail_results_, &err))
        {
            return CStatus(err);
        }
        CStatus st = publishResults(tail_results_);
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
