#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
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
}

std::vector<Detection> YoloPostprocessor::run(const cv::Mat &orig,
                                              ov::Tensor &out,
                                              const LetterBoxInfo &lb,
                                              const AppConfig &cfg,
                                              const std::function<bool(int)> &class_filter) const
{
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<bool> has_keypoints;
    std::vector<std::array<cv::Point2f, 4>> keypoints;
    auto shape = out.get_shape();
    const float *data = out.data<float>();
    if (shape.size() != 3)
    {
        return {};
    }

    const bool channel_first = (shape[1] < shape[2]);
    const int C = static_cast<int>(channel_first ? shape[1] : shape[2]);
    const int N = static_cast<int>(channel_first ? shape[2] : shape[1]);
    boxes.reserve(static_cast<size_t>(N));
    class_ids.reserve(static_cast<size_t>(N));
    scores.reserve(static_cast<size_t>(N));
    has_keypoints.reserve(static_cast<size_t>(N));
    keypoints.reserve(static_cast<size_t>(N));

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
        boxes.emplace_back(left, top, right - left, bottom - top);
        class_ids.push_back(cls);
        scores.push_back(score);
        has_keypoints.push_back(has_kpts);
        keypoints.push_back(OrderCorners(kpts));
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
                {
                    ++in01;
                }
            }
            float ratio = static_cast<float>(in01) / static_cast<float>(sample_n);
            if (ratio >= 0.98f)
            {
                ++inferred_cls;
            }
            else
            {
                break;
            }
        }

        int cls_count = std::max(0, C - 4);
        if (inferred_cls > 0)
        {
            cls_count = inferred_cls;
        }
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
            {
                continue;
            }
            const auto kpt_result = decode_keypoints(4 + use_cls, i);
            decode_box(cx, cy, w, h, best_cls, best_score, kpt_result.first, kpt_result.second);
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
            {
                continue;
            }
            const auto kpt_result = decode_keypoints(4 + use_cls, i);
            decode_box(cx, cy, w, h, best_cls, best_score, kpt_result.first, kpt_result.second);
        }
    }

    std::vector<int> candidate_indices;
    candidate_indices.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i)
    {
        if (!class_filter || class_filter(class_ids[i]))
        {
            candidate_indices.push_back(static_cast<int>(i));
        }
    }

    const int fast_nms_topk = cfg.fast_nms_topk;
    if (fast_nms_topk > 0 && static_cast<int>(candidate_indices.size()) > fast_nms_topk)
    {
        const auto topk_end = candidate_indices.begin() + fast_nms_topk;
        std::partial_sort(candidate_indices.begin(), topk_end, candidate_indices.end(), [&](int lhs, int rhs) {
            return scores[static_cast<std::size_t>(lhs)] > scores[static_cast<std::size_t>(rhs)];
        });
        candidate_indices.erase(topk_end, candidate_indices.end());
    }

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    nms_boxes.reserve(candidate_indices.size());
    nms_scores.reserve(candidate_indices.size());
    for (int idx : candidate_indices)
    {
        nms_boxes.push_back(boxes[static_cast<std::size_t>(idx)]);
        nms_scores.push_back(scores[static_cast<std::size_t>(idx)]);
    }

    std::vector<int> local_keep;
    cv::dnn::NMSBoxes(nms_boxes, nms_scores, conf_thres_, nms_thres_, local_keep);

    std::vector<Detection> dets;
    dets.reserve(local_keep.size());
    for (int idx : local_keep)
    {
        const int original_idx = candidate_indices[static_cast<std::size_t>(idx)];
        Detection d;
        d.box = boxes[static_cast<std::size_t>(original_idx)];
        d.class_id = class_ids[static_cast<std::size_t>(original_idx)];
        d.confidence = scores[static_cast<std::size_t>(original_idx)];
        d.has_keypoints = has_keypoints[static_cast<std::size_t>(original_idx)];
        d.keypoints = keypoints[static_cast<std::size_t>(original_idx)];
        dets.push_back(d);
    }
    return dets;
}
