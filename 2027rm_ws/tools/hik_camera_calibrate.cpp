#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "camera_node.hpp"
#include "logging.hpp"

namespace {

struct Options
{
    int board_cols = 9;
    int board_rows = 6;
    double square_size_m = 0.024;
    int samples = 25;
    std::string output = "2027rm_ws/config/camera_info.yaml";
    std::string camera_name = "narrow_stereo";
    std::string frame_id = "narrow_stereo_optical";
    std::string preview_dir;
};

void PrintUsage(const char *argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --board-cols N       Inner chessboard corners along width (default 9)\n"
        << "  --board-rows N       Inner chessboard corners along height (default 6)\n"
        << "  --square-size M      Chessboard square size in meters (default 0.024)\n"
        << "  --samples N          Number of accepted live samples (default 25)\n"
        << "  --output PATH        Output camera_info.yaml path\n"
        << "  --camera-name NAME   camera_name field (default narrow_stereo)\n"
        << "  --frame-id NAME      frame_id field (default narrow_stereo_optical)\n"
        << "  --preview-dir DIR    Save accepted corner preview images\n"
        << "  -h, --help           Show this help\n\n"
        << "Live controls: SPACE=save detected board, q/ESC=finish.\n";
}

bool ParseArgs(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help")
        {
            PrintUsage(argv[0]);
            return false;
        }
        if (arg == "--board-cols")
        {
            opt.board_cols = std::stoi(need_value("--board-cols"));
        }
        else if (arg == "--board-rows")
        {
            opt.board_rows = std::stoi(need_value("--board-rows"));
        }
        else if (arg == "--square-size")
        {
            opt.square_size_m = std::stod(need_value("--square-size"));
        }
        else if (arg == "--samples")
        {
            opt.samples = std::stoi(need_value("--samples"));
        }
        else if (arg == "--output")
        {
            opt.output = need_value("--output");
        }
        else if (arg == "--camera-name")
        {
            opt.camera_name = need_value("--camera-name");
        }
        else if (arg == "--frame-id")
        {
            opt.frame_id = need_value("--frame-id");
        }
        else if (arg == "--preview-dir")
        {
            opt.preview_dir = need_value("--preview-dir");
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opt.board_cols <= 0 || opt.board_rows <= 0 || opt.square_size_m <= 0.0 || opt.samples <= 0)
    {
        throw std::runtime_error("board size, square size, and samples must be positive");
    }
    return true;
}

std::vector<cv::Point3f> MakeObjectPoints(const cv::Size &board_size, double square_size_m)
{
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<std::size_t>(board_size.width * board_size.height));
    for (int y = 0; y < board_size.height; ++y)
    {
        for (int x = 0; x < board_size.width; ++x)
        {
            points.emplace_back(static_cast<float>(x * square_size_m),
                                static_cast<float>(y * square_size_m),
                                0.0f);
        }
    }
    return points;
}

bool FindCorners(const cv::Mat &bgr, const cv::Size &board_size, std::vector<cv::Point2f> &corners)
{
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    const int sb_flags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
    if (cv::findChessboardCornersSB(gray, board_size, corners, sb_flags))
    {
        return true;
    }

    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH |
                      cv::CALIB_CB_NORMALIZE_IMAGE |
                      cv::CALIB_CB_FAST_CHECK;
    if (!cv::findChessboardCorners(gray, board_size, corners, flags))
    {
        return false;
    }

    const cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, 1e-4);
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    return true;
}

std::string MatrixDataYaml(const cv::Mat &mat, int indent)
{
    std::ostringstream oss;
    const std::string pad(static_cast<std::size_t>(indent), ' ');
    oss << pad << "data: [\n";
    for (int r = 0; r < mat.rows; ++r)
    {
        oss << pad << "  ";
        for (int c = 0; c < mat.cols; ++c)
        {
            oss << cv::format("%.10g", mat.at<double>(r, c));
            if (r != mat.rows - 1 || c != mat.cols - 1)
            {
                oss << ", ";
            }
        }
        oss << "\n";
    }
    oss << pad << "]";
    return oss.str();
}

void WriteCameraInfo(const Options &opt,
                     const cv::Size &image_size,
                     const cv::Mat &camera_matrix,
                     const cv::Mat &dist_coeffs,
                     const cv::Mat &new_camera_matrix)
{
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    const cv::Mat flat_dist = dist_coeffs.reshape(1, 1);
    for (int i = 0; i < std::min(5, flat_dist.cols); ++i)
    {
        dist.at<double>(0, i) = flat_dist.at<double>(0, i);
    }

    const cv::Mat rectification = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat projection = cv::Mat::zeros(3, 4, CV_64F);
    new_camera_matrix.copyTo(projection(cv::Rect(0, 0, 3, 3)));

    const std::filesystem::path out_path(opt.output);
    if (!out_path.parent_path().empty())
    {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream fout(opt.output, std::ios::trunc);
    if (!fout.good())
    {
        throw std::runtime_error("failed to open output: " + opt.output);
    }

    fout << "image_width: " << image_size.width << "\n";
    fout << "image_height: " << image_size.height << "\n";
    fout << "camera_name: " << opt.camera_name << "\n";
    fout << "frame_id: " << opt.frame_id << "\n";
    fout << "camera_matrix:\n";
    fout << "  rows: 3\n";
    fout << "  cols: 3\n";
    fout << MatrixDataYaml(camera_matrix, 2) << "\n";
    fout << "distortion_model: plumb_bob\n";
    fout << "distortion_coefficients:\n";
    fout << "  rows: 1\n";
    fout << "  cols: 5\n";
    fout << MatrixDataYaml(dist, 2) << "\n";
    fout << "rectification_matrix:\n";
    fout << "  rows: 3\n";
    fout << "  cols: 3\n";
    fout << MatrixDataYaml(rectification, 2) << "\n";
    fout << "projection_matrix:\n";
    fout << "  rows: 3\n";
    fout << "  cols: 4\n";
    fout << MatrixDataYaml(projection, 2) << "\n";
}

double ComputeReprojectionRms(const std::vector<std::vector<cv::Point3f>> &object_points,
                              const std::vector<std::vector<cv::Point2f>> &image_points,
                              const cv::Mat &camera_matrix,
                              const cv::Mat &dist_coeffs,
                              const std::vector<cv::Mat> &rvecs,
                              const std::vector<cv::Mat> &tvecs)
{
    double total_error_sq = 0.0;
    std::size_t total_points = 0;
    for (std::size_t i = 0; i < object_points.size(); ++i)
    {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs, projected);
        const double err = cv::norm(image_points[i], projected, cv::NORM_L2);
        total_error_sq += err * err;
        total_points += projected.size();
    }
    return std::sqrt(total_error_sq / static_cast<double>(total_points));
}

}  // namespace

int main(int argc, char **argv)
{
    Options opt;
    try
    {
        if (!ParseArgs(argc, argv, opt))
        {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        PrintUsage(argv[0]);
        return 2;
    }

    app::logging::InitAsyncLogging();

    HikCameraNode camera;
    std::string error;
    if (!camera.init(&error))
    {
        std::cerr << "Camera init failed: " << error << "\n";
        return 1;
    }

    const cv::Size board_size(opt.board_cols, opt.board_rows);
    const std::vector<cv::Point3f> object_template = MakeObjectPoints(board_size, opt.square_size_m);
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    object_points.reserve(static_cast<std::size_t>(opt.samples));
    image_points.reserve(static_cast<std::size_t>(opt.samples));

    if (!opt.preview_dir.empty())
    {
        std::filesystem::create_directories(opt.preview_dir);
    }

    std::cout << "Hik camera calibration started.\n";
    std::cout << "Controls: SPACE=save detected board, q/ESC=finish.\n";

    cv::Mat frame;
    cv::Size image_size;
    while (static_cast<int>(object_points.size()) < opt.samples)
    {
        if (!camera.grab(frame, nullptr, &error))
        {
            std::cerr << "Grab failed: " << error << "\n";
            continue;
        }

        image_size = frame.size();
        std::vector<cv::Point2f> corners;
        const bool found = FindCorners(frame, board_size, corners);

        cv::Mat display = frame.clone();
        if (found)
        {
            cv::drawChessboardCorners(display, board_size, corners, found);
        }
        const std::string label =
            "samples " + std::to_string(object_points.size()) + "/" + std::to_string(opt.samples) +
            "  SPACE save  q finish";
        cv::putText(display,
                    label,
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                    2,
                    cv::LINE_AA);
        cv::imshow("hik camera calibration", display);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q')
        {
            break;
        }
        if (key == ' ' && found)
        {
            object_points.push_back(object_template);
            image_points.push_back(corners);
            std::cout << "[ok] captured sample " << object_points.size() << "/" << opt.samples << "\n";

            if (!opt.preview_dir.empty())
            {
                const std::filesystem::path path =
                    std::filesystem::path(opt.preview_dir) /
                    ("hik_" + cv::format("%03zu", object_points.size()) + ".jpg");
                cv::imwrite(path.string(), display);
            }
        }
    }

    camera.shutdown();
    cv::destroyAllWindows();

    if (object_points.size() < 8)
    {
        std::cerr << "Need at least 8 valid samples, got " << object_points.size() << ".\n";
        return 1;
    }

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    const double rms = cv::calibrateCamera(object_points,
                                           image_points,
                                           image_size,
                                           camera_matrix,
                                           dist_coeffs,
                                           rvecs,
                                           tvecs);
    cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix,
                                                              dist_coeffs,
                                                              image_size,
                                                              0.0,
                                                              image_size);
    const double reproj_rms = ComputeReprojectionRms(object_points,
                                                     image_points,
                                                     camera_matrix,
                                                     dist_coeffs,
                                                     rvecs,
                                                     tvecs);

    try
    {
        WriteCameraInfo(opt, image_size, camera_matrix, dist_coeffs, new_camera_matrix);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::cout << "\nSamples used: " << object_points.size() << "\n";
    std::cout << "Image size: " << image_size.width << "x" << image_size.height << "\n";
    std::cout << "OpenCV calibration RMS: " << rms << " px\n";
    std::cout << "Mean reprojection RMS: " << reproj_rms << " px\n";
    std::cout << "Wrote: " << std::filesystem::absolute(opt.output) << "\n";
    return 0;
}
