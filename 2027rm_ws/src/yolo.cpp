#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "camera_node.hpp"
#include "yolo_app.hpp"

using namespace CGraph;

static const char *FRAME_TOPIC = "rm/frame/topic";
static const char *RESULT_TOPIC = "rm/result/topic";

static AppConfig g_cfg = {
    "model/last.xml",
    "CPU",
    0,
    0.25f,
    0.25f,
    4,
    2};

static std::atomic<bool> g_stop(false);

struct FrameMParam : public GMessageParam
{
    cv::Mat frame;
    uint64_t frame_id;
    int64_t ts_ms;
};

struct ResultMParam : public GMessageParam
{
    cv::Mat vis;
    uint64_t frame_id;
    int infer_id = 0;
    double latency_ms = 0.0;
    int det_count = 0;
};

struct Detection
{
    cv::Rect box;
    int class_id = -1;
    float confidence = 0.0f;
};

static int64_t NowMs()
{
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
};

static float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static float ToProb(float x)
{
    if (x >= 0.0f && x <= 1.0f)
    {
        return x;
    }
    return Sigmoid(x);
}

static std::string ResolveModelPath(const std::string &model_path)
{
    const std::vector<std::string> candidates = {
        model_path,
        std::string("2027rm_ws/") + model_path,
        std::string("../2027rm_ws/") + model_path,
        std::string("../") + model_path};

    for (const auto &path : candidates)
    {
        std::ifstream fin(path);
        if (fin.good())
        {
            return path;
        }
    }
    return model_path;
}

class YoloOpenvino
{
public:
    bool init(const std::string &model_path, const std::string &device, int num_classes, float conf_thres, float nms_thres, std::string *error)
    {
        try
        {
            num_classes_ = num_classes;
            conf_thres_ = conf_thres;
            nms_thres_ = nms_thres;
            const std::string resolved_model_path = ResolveModelPath(model_path);
            auto model = ov::Core().read_model(resolved_model_path);
            compiled = ov::Core().compile_model(model, device);
            request_ = compiled.create_infer_request();
            input_port_ = compiled.input();
            output_port_ = compiled.output();
            auto in_shape = input_port_.get_shape();
            if (in_shape.size() != 4 || in_shape[1] != 3)
            {
                if (error)
                {
                    *error = "Unsupported input shape, expected [N,3,H,W]";
                }
                return false;
            }
            input_h_ = static_cast<int>(in_shape[2]);
            input_w_ = static_cast<int>(in_shape[3]);
            input_data_.resize(static_cast<size_t>(3 * input_h_ * input_w_));
            return true;
        }
        catch (const std::exception &e)
        {
            if (error)
            {
                *error = e.what();
            }
            return false;
        }
    }

    bool infer(const cv::Mat &frame, cv::Mat &vis, int &det_count, std::string *error)
    {
        if (frame.empty())
        {
            if (error)
            {
                *error = "Input frame is empty";
            }
            return false;
        }
        preprocess(frame);
        ov::Tensor in_tensor(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(input_h_), static_cast<size_t>(input_w_)}, input_data_.data());
        request_.set_input_tensor(in_tensor);
        request_.infer();
        ov::Tensor out = request_.get_output_tensor();
        std::vector<Detection> dets = postprocess(frame, out);
        vis = frame;
        for (const auto &d : dets)
        {
            cv::rectangle(vis, d.box, cv::Scalar(0, 255, 0), 2);
            std::string text = "id=" + std::to_string(d.class_id) + " conf=" + cv::format("%.2f", d.confidence);
            int ty = std::max(0, d.box.y - 5);
            cv::putText(vis, text, cv::Point(d.box.x, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
        det_count = static_cast<int>(dets.size());
        return true;
    }

private:
    struct LetterBoxInfo
    {
        float scale = 1.0f;
        int pad_w = 0;
        int pad_h = 0;
    };

    cv::Mat letterbox(const cv::Mat &src, LetterBoxInfo &lb) const
    {
        int src_w = src.cols;
        int src_h = src.rows;
        lb.scale = std::min(static_cast<float>(input_w_) / src_w, static_cast<float>(input_h_) / src_h);
        int nw = static_cast<int>(std::round(src_w * lb.scale));
        int nh = static_cast<int>(std::round(src_h * lb.scale));
        lb.pad_w = (input_w_ - nw) / 2;
        lb.pad_h = (input_h_ - nh) / 2;
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(nw, nh));
        cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(out(cv::Rect(lb.pad_w, lb.pad_h, nw, nh)));
        return out;
    };

    void preprocess(const cv::Mat &bgr)
    {
        cv::Mat lb_img = letterbox(bgr, lb_);
        cv::Mat rgb;
        cv::cvtColor(lb_img, rgb, cv::COLOR_BGR2RGB);
        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3, 1.0f / 255.0f);
        const int H = input_h_;
        const int W = input_w_;
        for (int y = 0; y < H; ++y)
        {
            const cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x)
            {
                input_data_[0 * H * W + y * W + x] = row[x][0];
                input_data_[1 * H * W + y * W + x] = row[x][1];
                input_data_[2 * H * W + y * W + x] = row[x][2];
            }
        }
    }
    std::vector<Detection> postprocess(const cv::Mat &orig, ov::Tensor &out) const
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> scores;
        auto shape = out.get_shape();
        const float *data = out.data<float>();
        if (shape.size() != 3)
            return {};

        const bool channel_first = (shape[1] < shape[2]);
        const int C = static_cast<int>(channel_first ? shape[1] : shape[2]);
        const int N = static_cast<int>(channel_first ? shape[2] : shape[1]);

        auto at = [&](int c, int i) -> float
        {
            if (channel_first)
            {
                return data[c * N + i];
            }
            return data[i * C + c];
        };

        auto decode_box = [&](float cx, float cy, float w, float h, int cls, float score)
        {
            float x1 = cx - 0.5f * w;
            float y1 = cy - 0.5f * h;
            float x2 = cx + 0.5f * w;
            float y2 = cy + 0.5f * h;
            x1 = (x1 - lb_.pad_w) / lb_.scale;
            y1 = (y1 - lb_.pad_h) / lb_.scale;
            x2 = (x2 - lb_.pad_w) / lb_.scale;
            y2 = (y2 - lb_.pad_h) / lb_.scale;
            int left = std::max(0, std::min(static_cast<int>(std::round(x1)), orig.cols - 1));
            int top = std::max(0, std::min(static_cast<int>(std::round(y1)), orig.rows - 1));
            int right = std::max(0, std::min(static_cast<int>(std::round(x2)), orig.cols - 1));
            int bottom = std::max(0, std::min(static_cast<int>(std::round(y2)), orig.rows - 1));
            if (right <= left || bottom <= top)
                return;
            boxes.emplace_back(left, top, right - left, bottom - top);
            class_ids.push_back(cls);
            scores.push_back(score);
        };

        // This model outputs decoded boxes as [cx, cy, w, h], then class probabilities,
        // then extra attributes (e.g. keypoints). We only use class channels.
        const std::vector<int> strides = {8, 16, 32};
        int expected_points = 0;
        for (int s : strides)
        {
            expected_points += (input_h_ / s) * (input_w_ / s);
        }

        if (C >= 6 && N == expected_points)
        {
            int inferred_cls = 0;
            const int sample_n = std::min(N, 256);
            for (int c = 4; c < C; ++c)
            {
                int in01 = 0;
                for (int i = 0; i < sample_n; ++i)
                {
                    float v = at(c, i);
                    if (v >= 0.0f && v <= 1.0f)
                        ++in01;
                }
                float ratio = static_cast<float>(in01) / static_cast<float>(sample_n);
                if (ratio >= 0.98f)
                    ++inferred_cls;
                else
                    break;
            }

            int cls_count = std::max(0, C - 4);
            if (inferred_cls > 0)
                cls_count = inferred_cls;
            int use_cls = (num_classes_ > 0) ? std::min(num_classes_, cls_count) : cls_count;

            for (int i = 0; i < N; ++i)
            {
                float cx = at(0, i);
                float cy = at(1, i);
                float w = at(2, i);
                float h = at(3, i);

                int best_cls = -1;
                float best_score = 0.0f;
                for (int c = 0; c < use_cls; ++c)
                {
                    float s = ToProb(at(4 + c, i));
                    if (s > best_score)
                    {
                        best_score = s;
                        best_cls = c;
                    }
                }

                if (best_score < conf_thres_)
                    continue;
                decode_box(cx, cy, w, h, best_cls, best_score);
            }
        }
        else
        {
            int cls_count = std::max(0, C - 4);
            int use_cls = (num_classes_ > 0) ? std::min(num_classes_, cls_count) : cls_count;
            for (int i = 0; i < N; ++i)
            {
                float cx = at(0, i);
                float cy = at(1, i);
                float w = at(2, i);
                float h = at(3, i);
                int best_cls = -1;
                float best_score = 0.0f;
                for (int c = 0; c < use_cls; ++c)
                {
                    float s = ToProb(at(4 + c, i));
                    if (s > best_score)
                    {
                        best_score = s;
                        best_cls = c;
                    }
                }
                if (best_score < conf_thres_)
                    continue;
                decode_box(cx, cy, w, h, best_cls, best_score);
            }
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres_, nms_thres_, keep);
        std::vector<Detection> dets;
        dets.reserve(keep.size());
        for (int idx : keep)
        {
            Detection d;
            d.box = boxes[idx];
            d.class_id = class_ids[idx];
            d.confidence = scores[idx];
            dets.push_back(d);
        }
        return dets;
    }

private:
    ov::Core core_;
    ov::CompiledModel compiled;
    ov::InferRequest request_;
    ov::Output<const ov::Node> input_port_;
    ov::Output<const ov::Node> output_port_;
    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 80;
    float conf_thres_ = 0.25f;
    float nms_thres_ = 0.45f;
    mutable LetterBoxInfo lb_;
    std::vector<float> input_data_;
};
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
            return CStatus(err);
        frame_id_ = 0;
        return CStatus();
    }
    CStatus run() override
    {
        while (!g_stop.load())
        {
            cv::Mat frame;
            std::string err;
            if (!camera_.grab(frame, &err))
            {
                return CStatus(err);
            }
            std::shared_ptr<FrameMParam> msg(new FrameMParam());
            msg->frame = std::move(frame);
            msg->frame_id = ++frame_id_;
            msg->ts_ms = NowMs();
            CStatus st = CGRAPH_SEND_MPARAM(FrameMParam, FRAME_TOPIC, msg, GMessagePushStrategy::WAIT);
            if (st.isErr())
                return st;
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

private:
    static int parseInferId(const std::string &name)
    {
        if (name.find("infer_2") != std::string::npos)
            return 2;
        return -1;
    }

public:
    CStatus init() override
    {
        std::string err;
        if (!yolo_.init(g_cfg.model_path, g_cfg.device, g_cfg.num_classes, g_cfg.conf_thres, g_cfg.nms_thres, &err))
        {
            return CStatus(err);
        }
        return CStatus();
    }
    CStatus run() override
    {
        const int infer_id = parseInferId(this->getName());
        while (!g_stop.load())
        {
            std::shared_ptr<FrameMParam> in = nullptr;
            CStatus st = CGRAPH_RECV_MPARAM_WITH_TIMEOUT(FrameMParam, FRAME_TOPIC, in, 1000);
            if (st.isErr())
            {
                if (g_stop.load())
                    break;
                continue;
            }
            if (!in || in->frame.empty())
                continue;
            cv::Mat vis;
            int det_count = 0;
            std::string err;
            if (!yolo_.infer(in->frame, vis, det_count, &err))
            {
                return CStatus(err);
            }
            std::shared_ptr<ResultMParam> out(new ResultMParam());
            out->vis = vis;
            out->frame_id = in->frame_id;
            out->infer_id = infer_id;
            out->latency_ms = static_cast<double>(NowMs() - in->ts_ms);
            out->det_count = det_count;
            st = CGRAPH_SEND_MPARAM(ResultMParam, RESULT_TOPIC, out, GMessagePushStrategy::WAIT);
            if (st.isErr())
                return st;
        }
        return CStatus();
    }
};

class DisplayNode : public GNode {
public:
    CStatus init() override {
        fps_start_ = std::chrono::steady_clock::now();
        count_ = 0;
        fps_ = 0.0;
        return CStatus();
    }

    CStatus run() override {
        while (!g_stop.load()) {
            std::shared_ptr<ResultMParam> r = nullptr;
            CStatus st = CGRAPH_RECV_MPARAM_WITH_TIMEOUT(ResultMParam, RESULT_TOPIC, r, 1000);
            if (st.isErr()) {
                if (g_stop.load()) break;
                continue;
            }
            if (!r || r->vis.empty()) continue;
            ++count_;
            auto now = std::chrono::steady_clock::now();
            auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start_).count();
            if (dt_ms >= 1000) {
                fps_ = count_ * 1000.0 / static_cast<double>(dt_ms);
                count_ = 0;
                fps_start_ = now;
            }
            cv::Mat &show = r->vis;
            std::string t1 = "frame=" + std::to_string(r->frame_id) + " infer=" + std::to_string(r->infer_id) + " det=" + std::to_string(r->det_count);
            std::string t2 = "fps=" + std::to_string(static_cast<int>(fps_ + 0.5));
            std::string t3 = "latency=" + std::to_string(static_cast<int>(r->latency_ms + 0.5)) + "ms";
            cv::putText(show, t1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            cv::putText(show, t2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::putText(show, t3, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
            cv::imshow("CGraph YOLO OpenVINO", show);
            int key = cv::waitKey(1);
            if (key == 27) {
                g_stop.store(true);
                return CStatus("user stopped by ESC");
            }
        }
        return CStatus();
    }

    CStatus destroy() override {
        cv::destroyAllWindows();
        return CStatus();
    }

private:
    std::chrono::steady_clock::time_point fps_start_;
    int count_ = 0;
    double fps_ = 0.0;
};

void ParseAppArgs(int argc, char** argv) {
    if (argc >= 2) g_cfg.model_path = argv[1];
    if (argc >= 3) g_cfg.device = argv[2];
    if (argc >= 4) g_cfg.num_classes = std::stoi(argv[3]);
    if (argc >= 5) g_cfg.conf_thres = std::stof(argv[4]);
    if (argc >= 6) g_cfg.nms_thres = std::stof(argv[5]);
    if (argc >= 7) g_cfg.thread_num = std::max(2, std::stoi(argv[6]));
    if (argc >= 8) g_cfg.infer_workers = std::max(1, std::min(2, std::stoi(argv[7])));
}

const AppConfig& GetAppConfig() {
    return g_cfg;
}

CStatus RegisterYoloPipelineElements(CGraph::GPipeline* const &pipeline) {
    GElementPtr cam = nullptr;
    GElementPtr infer1 = nullptr;
    GElementPtr infer2 = nullptr;
    GElementPtr display = nullptr;
    CStatus st;
    st += pipeline->registerGElement<CameraPubNode>(&cam, {}, "camera_pub");
    st += pipeline->registerGElement<YoloInferNode>(&infer1, {}, "infer_1");
    if (g_cfg.infer_workers >= 2) {
        st += pipeline->registerGElement<YoloInferNode>(&infer2, {}, "infer_2");
    }
    st += pipeline->registerGElement<DisplayNode>(&display, {}, "display");
    return st;
}

void InitYoloMessageTopics() {
    CGRAPH_CREATE_MESSAGE_TOPIC(FrameMParam, FRAME_TOPIC, 2);
    CGRAPH_CREATE_MESSAGE_TOPIC(ResultMParam, RESULT_TOPIC, 2);
}

void ShutdownYoloApp() {
    g_stop.store(true);
    CGRAPH_CLEAR_MESSAGES()
}