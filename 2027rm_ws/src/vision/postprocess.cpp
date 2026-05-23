#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/dnn.hpp>

namespace {

float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

float ToProb(float x)
{
    if (x >= 0.0f && x <= 1.0f)
    {
        return x;
    }
    return Sigmoid(x);
}

}  // namespace

YoloPostprocessor::YoloPostprocessor(int input_w, int input_h, int num_classes, float conf_thres, float nms_thres)
    : input_w_(input_w),
      input_h_(input_h),
      num_classes_(num_classes),
      conf_thres_(conf_thres),
      nms_thres_(nms_thres)
{
    output_layout_.shape.reserve(3);
    const int expected_points = (input_h_ / 8) * (input_w_ / 8) +
                                (input_h_ / 16) * (input_w_ / 16) +
                                (input_h_ / 32) * (input_w_ / 32);
    reserveScratch(static_cast<std::size_t>(std::max(0, expected_points)));
}

void YoloPostprocessor::reserveScratch(std::size_t points)
{
    Scratch &scratch = scratch_;
    scratch.boxes.reserve(points);
    scratch.class_ids.reserve(points);
    scratch.scores.reserve(points);
    scratch.has_keypoints.reserve(points);
    scratch.keypoints.reserve(points);
    scratch.candidate_indices.reserve(points);
    scratch.nms_boxes.reserve(points);
    scratch.nms_scores.reserve(points);
    scratch.local_keep.reserve(points);
    scratch.keep.reserve(points);
    scratch.detections.reserve(points);
}

const YoloPostprocessor::OutputLayoutCache &YoloPostprocessor::resolveOutputLayout(const ov::Shape &shape, const float *data)
{
    if (output_layout_.valid && output_layout_.shape == shape)
    {
        return output_layout_;
    }

    output_layout_ = OutputLayoutCache{};
    output_layout_.shape.assign(shape.begin(), shape.end());
    if (shape.size() != 3)
    {
        return output_layout_;
    }

    output_layout_.channel_first = (shape[1] < shape[2]);
    output_layout_.channels = static_cast<int>(output_layout_.channel_first ? shape[1] : shape[2]);
    output_layout_.points = static_cast<int>(output_layout_.channel_first ? shape[2] : shape[1]);
    const int C = output_layout_.channels;
    const int N = output_layout_.points;
    auto at = [&](int c, int i) -> float {
        if (output_layout_.channel_first)
        {
            return data[c * N + i];
        }
        return data[i * C + c];
    };

    const int expected_points = (input_h_ / 8) * (input_w_ / 8) +
                                (input_h_ / 16) * (input_w_ / 16) +
                                (input_h_ / 32) * (input_w_ / 32);
    output_layout_.expected_anchor_points = (C >= 6 && N == expected_points);
    int cls_count = std::max(0, C - 4);
    if (output_layout_.expected_anchor_points)
    {
        int inferred_cls = 0;
        const int sample_n = std::min(N, 256);
        for (int c = 4; c < C; ++c)
        {
            int in01 = 0;
            for (int i = 0; i < sample_n; ++i)
            {
                const float v = at(c, i);
                if (v >= 0.0f && v <= 1.0f)
                {
                    ++in01;
                }
            }
            const float ratio = static_cast<float>(in01) / static_cast<float>(sample_n);
            if (ratio >= 0.98f)
            {
                ++inferred_cls;
            }
            else
            {
                break;
            }
        }
        if (inferred_cls > 0)
        {
            cls_count = inferred_cls;
        }
    }
    output_layout_.use_classes = (num_classes_ > 0) ? std::min(num_classes_, cls_count) : cls_count;
    output_layout_.valid = true;
    return output_layout_;
}

std::vector<Detection> &YoloPostprocessor::run(const cv::Mat &orig,
                                               ov::Tensor &out,
                                               const LetterBoxInfo &lb,
                                               const AppConfig &cfg,
                                               std::span<const uint8_t> class_allowed)
{
    const auto shape = out.get_shape();
    const float *data = out.data<float>();
    const OutputLayoutCache &layout = resolveOutputLayout(shape, data);
    if (!layout.valid)
    {
        scratch_.detections.clear();
        return scratch_.detections;
    }

    const bool channel_first = layout.channel_first;
    const int C = layout.channels;
    const int N = layout.points;
    const int use_cls = layout.use_classes;
    const bool filter_classes = !class_allowed.empty();
    Scratch &scratch = scratch_;
    scratch.boxes.clear();
    scratch.class_ids.clear();
    scratch.scores.clear();
    scratch.has_keypoints.clear();
    scratch.keypoints.clear();
    scratch.candidate_indices.clear();
    scratch.nms_boxes.clear();
    scratch.nms_scores.clear();
    scratch.local_keep.clear();
    scratch.keep.clear();
    scratch.detections.clear();
    scratch.boxes.reserve(static_cast<size_t>(N));
    scratch.class_ids.reserve(static_cast<size_t>(N));
    scratch.scores.reserve(static_cast<size_t>(N));
    scratch.has_keypoints.reserve(static_cast<size_t>(N));
    scratch.keypoints.reserve(static_cast<size_t>(N));

    auto at = [&](int c, int i) -> float {
        if (channel_first)
        {
            return data[c * N + i];
        }
        return data[i * C + c];
    };

    auto image_point_from_model = [&](float x, float y) -> cv::Point2f {
        x = (x - static_cast<float>(lb.pad_w)) / lb.scale;
        y = (y - static_cast<float>(lb.pad_h)) / lb.scale;
        x = std::max(0.0f, std::min(x, static_cast<float>(orig.cols - 1)));
        y = std::max(0.0f, std::min(y, static_cast<float>(orig.rows - 1)));
        return cv::Point2f(x, y);
    };

    auto decode_box = [&](float cx,
                          float cy,
                          float w,
                          float h,
                          int cls,
                          float score,
                          bool has_kpts,
                          const std::array<cv::Point2f, 4> &kpts) {
        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;
        const cv::Point2f pt1 = image_point_from_model(x1, y1);
        const cv::Point2f pt2 = image_point_from_model(x2, y2);
        x1 = pt1.x;
        y1 = pt1.y;
        x2 = pt2.x;
        y2 = pt2.y;
        int left = std::max(0, std::min(static_cast<int>(std::round(x1)), orig.cols - 1));
        int top = std::max(0, std::min(static_cast<int>(std::round(y1)), orig.rows - 1));
        int right = std::max(0, std::min(static_cast<int>(std::round(x2)), orig.cols - 1));
        int bottom = std::max(0, std::min(static_cast<int>(std::round(y2)), orig.rows - 1));
        if (right <= left || bottom <= top)
        {
            return;
        }
        scratch.boxes.emplace_back(left, top, right - left, bottom - top);
        scratch.class_ids.push_back(cls);
        scratch.scores.push_back(score);
        scratch.has_keypoints.push_back(static_cast<uint8_t>(has_kpts));
        scratch.keypoints.push_back(OrderCorners(kpts));
    };

    auto decode_keypoints = [&](int first_kpt_channel, int i) -> std::pair<bool, std::array<cv::Point2f, 4>> {
        std::array<cv::Point2f, 4> kpts = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(0.0f, 0.0f)};
        const int attr_count = C - first_kpt_channel;
        const int stride = (attr_count >= 12) ? 3 : ((attr_count >= 8) ? 2 : 0);
        if (stride == 0)
        {
            return {false, kpts};
        }

        bool valid = true;
        for (int k = 0; k < 4; ++k)
        {
            const int base = first_kpt_channel + k * stride;
            float x = at(base, i);
            float y = at(base + 1, i);
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                valid = false;
                break;
            }
            if (stride == 3)
            {
                const float kpt_score = ToProb(at(base + 2, i));
                if (kpt_score < 0.05f)
                {
                    valid = false;
                }
            }
            kpts[static_cast<std::size_t>(k)] = image_point_from_model(x, y);
        }

        return {valid, kpts};
    };

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
        {
            continue;
        }
        if (filter_classes &&
            (best_cls < 0 ||
             static_cast<std::size_t>(best_cls) >= class_allowed.size() ||
             class_allowed[static_cast<std::size_t>(best_cls)] == 0U))
        {
            continue;
        }
        const auto kpt_result = decode_keypoints(4 + use_cls, i);
        decode_box(cx, cy, w, h, best_cls, best_score, kpt_result.first, kpt_result.second);
    }

    scratch.candidate_indices.reserve(scratch.scores.size());
    for (std::size_t i = 0; i < scratch.scores.size(); ++i)
    {
        scratch.candidate_indices.push_back(static_cast<int>(i));
    }

    const int fast_nms_topk = cfg.fast_nms_topk;
    if (fast_nms_topk > 0 && static_cast<int>(scratch.candidate_indices.size()) > fast_nms_topk)
    {
        const auto topk_end = scratch.candidate_indices.begin() + fast_nms_topk;
        std::nth_element(scratch.candidate_indices.begin(), topk_end, scratch.candidate_indices.end(), [&](int lhs, int rhs) {
            return scratch.scores[static_cast<std::size_t>(lhs)] > scratch.scores[static_cast<std::size_t>(rhs)];
        });
        scratch.candidate_indices.erase(topk_end, scratch.candidate_indices.end());
    }

    scratch.nms_boxes.reserve(scratch.candidate_indices.size());
    scratch.nms_scores.reserve(scratch.candidate_indices.size());
    for (int idx : scratch.candidate_indices)
    {
        scratch.nms_boxes.push_back(scratch.boxes[static_cast<std::size_t>(idx)]);
        scratch.nms_scores.push_back(scratch.scores[static_cast<std::size_t>(idx)]);
    }

    cv::dnn::NMSBoxes(scratch.nms_boxes, scratch.nms_scores, conf_thres_, nms_thres_, scratch.local_keep);

    scratch.detections.reserve(scratch.local_keep.size());
    for (int idx : scratch.local_keep)
    {
        const int original_idx = scratch.candidate_indices[static_cast<std::size_t>(idx)];
        Detection d;
        d.box = scratch.boxes[static_cast<std::size_t>(original_idx)];
        d.class_id = scratch.class_ids[static_cast<std::size_t>(original_idx)];
        d.confidence = scratch.scores[static_cast<std::size_t>(original_idx)];
        d.has_keypoints = scratch.has_keypoints[static_cast<std::size_t>(original_idx)] != 0;
        d.keypoints = scratch.keypoints[static_cast<std::size_t>(original_idx)];
        scratch.detections.push_back(d);
    }
    return scratch.detections;
}
