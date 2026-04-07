#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "camera_node.hpp"

using namespace CGraph;

static const char *FRAME_TOPIC = "rm/frame/topic";
static const char *RESULT_TOPIC = "rm/result/topic";

struct AppConfig
{
    std::string model_path = "model/yolov5s.xml";
    std::string device = "CPU";
    int num_classes = 14;
    float conf_thres = 0.25f;
    float nms_thres = 0.25f;
    int thread_num = 4;
    int infer_workers = 2;
} g_cfg;

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
class YoloOpenvino
{
    public:
    bool  init(const std::string& model_path,const std::string&device,int num_classes, float conf_thres,float nms_thres,std::string*error)
    {
        try
        {
            num_classes_ = num_classes;
            conf_thres_ = conf_thres;
            nms_thres_ = nms_thres;
            auto model=ov::Core().read_model(model_path);
            compiled=ov::Core().compile_model(model,device);
            request_=compiled.create_infer_request();
            input_port_=compiled.input();
            output_port_=compiled.output();
            auto  in_shape=input_port_.get_shape();
            if(in_shape.size()!=4 || in_shape[1]!=3)
            {
                if(error)
                {
                    *error="Unsupported input shape, expected [N,3,H,W]";
                }
                return false;
            }
            input_h_=static_cast<int>(in_shape[2]);
            input_w_=static_cast<int>(in_shape[3]);
            input_data_.resize(static_cast<size_t>(3*input_h_*input_w_ ));
            return true;
        }
        catch(const std::exception& e)
        {
            if(error)
            {
                *error=e.what();
            }
            return false;
        }
        
    }

    bool infer(const cv::Mat& frame,cv::Mat&vis,int&det_count,std::string*error)
    {
        if(frame.empty())
        {
            if(error)
            {
                *error="Input frame is empty";
            }
            return false;
        }
        preprocess(frame);
        ov::Tensor in_tensor(ov::element::f32,ov::Shape{1,3,static_cast<size_t>(input_h_),static_cast<size_t>(input_w_)},input_data_.data());
        request_.set_input_tensor(in_tensor);
        request_.infer();
        ov::Tensor out=request_.get_output_tensor();
        std::vector<Detection>dets=postprocess(frame,out);
        vis=frame.clone();
        for(const auto&d:dets)
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
        float scale=1.0f;
        int pad_w=0;
        int pad_h=0;
    };

    cv::Mat letterbox(const cv::Mat& src,LetterBoxInfo& lb)const
    {
        int src_w=src.cols;
        int src_h=src.rows;
        lb.scale=std::min(static_cast<float>(input_w_)/src_w,static_cast<float>(input_h_)/src_h);
        int nw=static_cast<int>(std::round(src_w*lb.scale));
        int nh=static_cast<int>(std::round(src_h*lb.scale));
        lb.pad_w=(input_w_-nw)/2;
        lb.pad_h=(input_h_-nh)/2;
        cv::Mat resized;
        cv::resize(src,resized,cv::Size(nw,nh));
        cv::Mat out(input_h_,input_w_,CV_8UC3,cv::Scalar(114,114,114)); 
        resized.copyTo(out(cv::Rect(lb.pad_w,lb.pad_h,nw,nh)));
        return out;
    };

    void preprocess(const cv::Mat& bgr)
    {
        cv::Mat lb_img=letterbox(bgr,lb_);
        cv::Mat rgb;
        cv::cvtColor(lb_img,rgb,cv::COLOR_BGR2RGB);
        cv::Mat f32;
        rgb.convertTo(f32,CV_32FC3,1.0f/255.0f);
        const int H=input_h_;
        const int W=input_w_;
        for(int y=0;y<H;++y)
        {
            const cv::Vec3f*row =f32.ptr<cv::Vec3f>(y);
            for(int x=0;x<W;++x)
            {
                input_data_[0*H*W+y*W+x]=row[x][0];
                input_data_[1*H*W+y*W+x]=row[x][1];
                input_data_[2*H*W+y*W+x]=row[x][2];
            }
        }
    }
    std::vector<Detection>postprocess(const cv::Mat&orig,const ov::Tensor&out)const
    {
        std::vector<cv::Rect>boxes;
        std::vector<int>class_ids;
        std::vector<float>scores;
        auto shape=out.get_shape();
        const float*data=out.data<const float>();
        if(shape.size()!=3)return{};
        auto decode_box=[&](float cx,float cy,float w,float h,int cls,float score)
        {
            float x1=cx-0.5f*w;
            float y1=cy-0.5f*h;
            float x2=cx+0.5f*w;
            float y2=cy+0.5f*h;
            x1 = (x1 - lb_.pad_w) / lb_.scale;
            y1 = (y1 - lb_.pad_h) / lb_.scale;
            x2 = (x2 - lb_.pad_w) / lb_.scale;
            y2 = (y2 - lb_.pad_h) / lb_.scale;
            int left = std::max(0, std::min(static_cast<int>(std::round(x1)), orig.cols - 1));
            int top = std::max(0, std::min(static_cast<int>(std::round(y1)), orig.rows - 1));
            int right = std::max(0, std::min(static_cast<int>(std::round(x2)), orig.cols - 1));
            int bottom = std::max(0, std::min(static_cast<int>(std::round(y2)), orig.rows -1));
            if(right<=left||bottom<=top)return;
            boxes.emplace_back(left,top,right-left,bottom-top);
            class_ids.push_back(cls);
            scores.push_back(score);
        };
        if(shape[1]<shape[2])
        {
            int C=static_cast<int>(shape[1]);
            int N=static_cast<int>(shape[2]);
            int cls_count=std::max(0,C-4);
            int use_cls=(num_classes_>0)?std::min(num_classes_,cls_count):cls_count;
            for(int i=0;i<N;++i)
            {
                float cx=data[0*N+i];
                float cy=data[1*N+i];
                float w=data[2*N+i];
                float h=data[3*N+i];
                int best_cls=-1;
                float best_score=0.0f;
                for(int c=0;c<use_cls;++c)
                {
                    float s=data[(4+c)*N+i];
                    if(s>best_score)
                    {
                        best_score=s;
                        best_cls=c;
                    }
                }
                if(best_score<conf_thres_)continue;
                decode_box(cx,cy,w,h,best_cls,best_score);  
            }
        }
        else
        {
            int N=static_cast<int>(shape[1]);
            int C=static_cast<int>(shape[2]);
            int cls_count=std::max(0,C-4);
            int use_cls=(num_classes_>0)?std::min(num_classes_,cls_count):cls_count;
            for(int i=0;i<N;++i)
            {
                const float*p=data+i*C;
                float cx=p[0];
                float cy=p[1];
                float w=p[2];
                float h=p[3];
                int best_cls=-1;
                float best_score=0.0f;
                for(int c=0;c<use_cls;++c)
                {
                    float s=p[4+c];
                    if(s>best_score)
                    {
                        best_score=s;
                        best_cls=c;
                    }
                }
                if(best_score<conf_thres_)continue;
                decode_box(cx,cy,w,h,best_cls,best_score);  
            }
        }
        std::vector<int>keep;
        cv::dnn::NMSBoxes(boxes,scores,conf_thres_,nms_thres_,keep);
        std::vector<Detection>dets;
        dets.reserve(keep.size());
        for(int idx:keep)
        {
            Detection d;
            d.box=boxes[idx];
            d.class_id=class_ids[idx];
            d.confidence=scores[idx];
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
    int input_w_=640;
    int input_h_=640;
    int num_classes_=80;
    float conf_thres_=0.25f;
    float nms_thres_=0.45f;
    mutable LetterBoxInfo lb_;
    std::vector<float>input_data_;
};