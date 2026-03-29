#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "detector_node.hpp"

struct CameraIntrinsics
{
	double fx = 0.0;
};

struct LightBar
{
	cv::RotatedRect rect;
	float length_px = 0.0f;
	float width_px = 0.0f;
	float angle_deg = 0.0f;
	cv::Point2f center;
};

static float NormalizeAngle180(float angle)
{
	while (angle < 0.0f)
	{
		angle += 180.0f;
	}
	while (angle >= 180.0f)
	{
		angle -= 180.0f;
	}
	return angle;
}

static float AngleDiff180(float a, float b)
{
	float d = std::fabs(NormalizeAngle180(a) - NormalizeAngle180(b));
	if (d > 90.0f)
	{
		d = 180.0f - d;
	}
	return d;
}

static bool LoadFxFromYaml(const std::string &yaml_path, CameraIntrinsics &K)
{
	try
	{
		YAML::Node root = YAML::LoadFile(yaml_path);
		if (!root["camera_matrix"] || !root["camera_matrix"]["data"])
		{
			return false;
		}
		const YAML::Node data = root["camera_matrix"]["data"];
		if (!data.IsSequence() || data.size() < 1)
		{
			return false;
		}
		K.fx = data[0].as<double>();
		return K.fx > 0.0;
	}
	catch (const std::exception &)
	{
		return false;
	}
}

static float ComputePxPerCm(double fx, float depth_cm)
{
	if (fx <= 0.0 || depth_cm <= 1e-3f)
	{
		return 0.0f;
	}
	return static_cast<float>(fx / depth_cm);
}

static std::vector<LightBar> ExtractRedLightBars(const cv::Mat &bgr,
												 const float expected_len_px,
												 const float len_tol_ratio)
{
	cv::Mat hsv;
	cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask1, mask2, mask_hsv;
	cv::inRange(hsv, cv::Scalar(0, 120, 80), cv::Scalar(12, 255, 255), mask1);
	cv::inRange(hsv, cv::Scalar(165, 120, 80), cv::Scalar(179, 255, 255), mask2);
	mask_hsv = mask1 | mask2;

	std::vector<cv::Mat> bgr_ch;
	cv::split(bgr, bgr_ch);
	const cv::Mat &b = bgr_ch[0];
	const cv::Mat &g = bgr_ch[1];
	const cv::Mat &r = bgr_ch[2];

	cv::Mat rg_dom, rb_dom, r_bright, mask_rgb;
	cv::compare(r, g + 20, rg_dom, cv::CMP_GT);
	cv::compare(r, b + 20, rb_dom, cv::CMP_GT);
	cv::compare(r, 80, r_bright, cv::CMP_GT);
	mask_rgb = rg_dom & rb_dom & r_bright;

	cv::Mat mask = mask_hsv | mask_rgb;

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 7)));

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<LightBar> bars;
	bars.reserve(contours.size());

	const float len_min = std::max(8.0f, expected_len_px * (1.0f - len_tol_ratio));
	const float len_max = std::max(len_min + 2.0f, expected_len_px * (1.0f + len_tol_ratio));

	for (size_t i = 0; i < contours.size(); ++i)
	{
		const double area = cv::contourArea(contours[i]);
		if (area < 30.0)
		{
			continue;
		}

		cv::RotatedRect rr = cv::minAreaRect(contours[i]);
		float w = rr.size.width;
		float h = rr.size.height;
		float length = std::max(w, h);
		float width = std::min(w, h);
		if (width < 1.0f)
		{
			continue;
		}

		const float aspect = length / width;
		if (aspect < 2.5f)
		{
			continue;
		}

		if (length < len_min || length > len_max)
		{
			continue;
		}

		float angle = rr.angle;
		if (rr.size.width < rr.size.height)
		{
			angle += 90.0f;
		}

		LightBar bar;
		bar.rect = rr;
		bar.length_px = length;
		bar.width_px = width;
		bar.angle_deg = NormalizeAngle180(angle);
		bar.center = rr.center;
		bars.push_back(bar);
	}

	return bars;
}

static bool IsGoodPair(const LightBar &a,
					   const LightBar &b,
					   const float expected_gap_px,
					   const float gap_tol_ratio,
					   const float parallel_deg_th)
{
	const float angle_diff = AngleDiff180(a.angle_deg, b.angle_deg);
	if (angle_diff > parallel_deg_th)
	{
		return false;
	}

	const float len_ratio = std::max(a.length_px, b.length_px) / std::min(a.length_px, b.length_px);
	if (len_ratio > 1.35f)
	{
		return false;
	}

	const float y_diff = std::fabs(a.center.y - b.center.y);
	const float avg_len = 0.5f * (a.length_px + b.length_px);
	if (y_diff > 0.65f * avg_len)
	{
		return false;
	}

	const float center_dist = cv::norm(a.center - b.center);
	const float gap_min = expected_gap_px * (1.0f - gap_tol_ratio);
	const float gap_max = expected_gap_px * (1.0f + gap_tol_ratio);
	if (center_dist < gap_min || center_dist > gap_max)
	{
		return false;
	}

	const float gap_len_ratio = center_dist / std::max(1.0f, avg_len);
	if (gap_len_ratio < 1.2f || gap_len_ratio > 4.0f)
	{
		return false;
	}

	return true;
}

bool RedLightBarDetector::init(float depth_cm,
						   const std::string &camera_info_yaml,
						   float expected_len_cm,
						   float expected_gap_cm,
						   std::string *error)
{
	CameraIntrinsics K;
	if (!LoadFxFromYaml(camera_info_yaml, K))
	{
		if (error)
		{
			*error = "failed to load camera intrinsics";
		}
		return false;
	}

	const float px_per_cm = ComputePxPerCm(K.fx, depth_cm);
	if (px_per_cm <= 0.0f)
	{
		if (error)
		{
			*error = "invalid fx or depth";
		}
		return false;
	}

	expected_len_px_ = expected_len_cm * px_per_cm;
	expected_gap_px_ = expected_gap_cm * px_per_cm;
	initialized_ = true;
	return true;
}

bool RedLightBarDetector::detect(const cv::Mat &frame,
						 cv::Mat &vis,
						 bool *found,
						 std::string *error) const
{
	if (!initialized_)
	{
		if (error)
		{
			*error = "detector not initialized";
		}
		return false;
	}

	if (frame.empty())
	{
		if (error)
		{
			*error = "empty frame";
		}
		return false;
	}

	std::vector<LightBar> bars = ExtractRedLightBars(frame, expected_len_px_, len_tol_ratio_);
	vis = frame.clone();
	for (size_t i = 0; i < bars.size(); ++i)
	{
		cv::Point2f pts[4];
		bars[i].rect.points(pts);
		for (int k = 0; k < 4; ++k)
		{
			cv::line(vis, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 255, 255), 1);
		}
	}

	bool localFound = false;
	LightBar best_a;
	LightBar best_b;
	float best_score = 1e9f;
	for (size_t i = 0; i < bars.size(); ++i)
	{
		for (size_t j = i + 1; j < bars.size(); ++j)
		{
			if (!IsGoodPair(bars[i], bars[j], expected_gap_px_, gap_tol_ratio_, parallel_deg_th_))
			{
				continue;
			}

			const float center_dist = cv::norm(bars[i].center - bars[j].center);
			const float angle_diff = AngleDiff180(bars[i].angle_deg, bars[j].angle_deg);
			const float score = std::fabs(center_dist - expected_gap_px_) + 2.0f * angle_diff;
			if (score < best_score)
			{
				best_score = score;
				best_a = bars[i];
				best_b = bars[j];
				localFound = true;
			}
		}
	}

	if (localFound)
	{
		cv::line(vis, best_a.center, best_b.center, cv::Scalar(0, 255, 0), 2);
		cv::putText(vis, "Parallel red light bars found", cv::Point(20, 35),
						cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
	}
	else
	{
		cv::putText(vis, "No valid pair", cv::Point(20, 35),
						cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	}

	if (found)
	{
		*found = localFound;
	}

	return true;
}

#ifndef DETECTOR_NODE_LIBRARY
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: detector_app <image_or_video> [target_depth_cm] [camera_info_yaml]" << std::endl;
		std::cout << "Example: detector_app test.jpg 100 ../config/camera_info.yaml" << std::endl;
		return 1;
	}

	const std::string input_path = argv[1];
	const float depth_cm = (argc >= 3) ? std::stof(argv[2]) : 100.0f;
	const std::string camera_info_path = (argc >= 4) ? argv[3] : "../config/camera_info.yaml";

	RedLightBarDetector detector;
	std::string error;
	if (!detector.init(depth_cm, camera_info_path, 6.0f, 14.0f, &error))
	{
		std::cerr << "Detector init failed: " << error << std::endl;
		return 2;
	}

	cv::VideoCapture cap(input_path);
	const bool is_video = cap.isOpened();

	cv::Mat frame;
	if (!is_video)
	{
		frame = cv::imread(input_path);
		if (frame.empty())
		{
			std::cerr << "Cannot open input: " << input_path << std::endl;
			return 4;
		}
	}

	while (true)
	{
		if (is_video)
		{
			cap >> frame;
			if (frame.empty())
			{
				break;
			}
		}

		cv::Mat vis;
		bool found = false;
		if (!detector.detect(frame, vis, &found, &error))
		{
			std::cerr << "Detect failed: " << error << std::endl;
			continue;
		}

		const std::string info = "depth=" + std::to_string(static_cast<int>(depth_cm + 0.5f)) + "cm";
		cv::putText(vis, info, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.65,
					cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

		cv::imshow("Red Light Bar Detector", vis);
		const int key = cv::waitKey(is_video ? 1 : 0);
		if (key == 27)
		{
			break;
		}
		if (!is_video)
		{
			break;
		}
	}

	return 0;
}
#endif
