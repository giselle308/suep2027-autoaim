#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <CGraph.h>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

#include "message_pool.hpp"
#include "thread_affinity.hpp"
#include "yolo_app.hpp"
#include "yolo_common.hpp"

using namespace CGraph;

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kArmorTiltRad = 15.0 * kPi / 180.0;
constexpr int kStateDim = 11;
constexpr int kMeasDim = 4;
constexpr double kNisRejectThreshold = 13.28;  // chi-square 4 dof, 99%
constexpr double kNeesRejectThreshold = 24.73; // chi-square 11 dof, 99%
constexpr int kConsistencyWindow = 50;

using StateVec = std::array<double, kStateDim>;
using MeasVec = std::array<double, kMeasDim>;
using StateMat = std::array<std::array<double, kStateDim>, kStateDim>;
using MeasMat = std::array<std::array<double, kMeasDim>, kMeasDim>;
using CrossMat = std::array<std::array<double, kMeasDim>, kStateDim>;

enum class ArmorSlot
{
    FRONT = 0,
    LEFT = 1,
    BACK = 2,
    RIGHT = 3,
};

struct ArmorPose
{
    ArmorSlot slot = ArmorSlot::FRONT;
    cv::Vec3d center_m = cv::Vec3d(0.0, 0.0, 0.0);
    double yaw_rad = 0.0;
    double pitch_rad = kArmorTiltRad;
};

struct Geometry
{
    double front_radius_m = 0.22;
    double back_radius_m = 0.22;
    double left_radius_m = 0.20;
    double right_radius_m = 0.20;
    double height_diff_m = 0.0;
};

struct CkfUpdateStats
{
    bool accepted = false;
    double nis = std::numeric_limits<double>::quiet_NaN();
    double nees = std::numeric_limits<double>::quiet_NaN();
    bool nis_fail = false;
    bool nees_fail = false;
};

double WrapAngleRad(double angle)
{
    while (angle > kPi)
    {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi)
    {
        angle += 2.0 * kPi;
    }
    return angle;
}

const char *SlotName(ArmorSlot slot)
{
    switch (slot)
    {
    case ArmorSlot::FRONT:
        return "FRONT";
    case ArmorSlot::LEFT:
        return "LEFT";
    case ArmorSlot::BACK:
        return "BACK";
    case ArmorSlot::RIGHT:
        return "RIGHT";
    }
    return "UNKNOWN";
}

double SlotYawOffsetRad(ArmorSlot slot)
{
    switch (slot)
    {
    case ArmorSlot::FRONT:
        return 0.0;
    case ArmorSlot::LEFT:
        return kPi * 0.5;
    case ArmorSlot::BACK:
        return kPi;
    case ArmorSlot::RIGHT:
        return -kPi * 0.5;
    }
    return 0.0;
}

cv::Vec3d RotateCameraY(const cv::Vec3d &v, double yaw_rad)
{
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    return cv::Vec3d(c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]);
}

cv::Vec3d SlotOffsetBody(ArmorSlot slot, const Geometry &g)
{
    switch (slot)
    {
    case ArmorSlot::FRONT:
        return cv::Vec3d(0.0, 0.0, -g.front_radius_m);
    case ArmorSlot::BACK:
        return cv::Vec3d(0.0, 0.0, g.back_radius_m);
    case ArmorSlot::LEFT:
        return cv::Vec3d(-g.left_radius_m, g.height_diff_m, 0.0);
    case ArmorSlot::RIGHT:
        return cv::Vec3d(g.right_radius_m, g.height_diff_m, 0.0);
    }
    return cv::Vec3d(0.0, 0.0, 0.0);
}

double ArmorYawRadFromRvec(const cv::Vec3d &rvec_input)
{
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << rvec_input[0], rvec_input[1], rvec_input[2]);
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    return std::atan2(rotation_matrix.at<double>(0, 0),
                      rotation_matrix.at<double>(2, 0));
}

double ArmorYawRadFromPnp(const PnpResultMParam &pnp)
{
    return ArmorYawRadFromRvec(pnp.rvec);
}

double ArmorYawRadFromPose(const PnpResultMParam::ArmorPose &pose)
{
    return ArmorYawRadFromRvec(pose.rvec);
}

double Distance3d(const cv::Vec3d &a, const cv::Vec3d &b)
{
    const cv::Vec3d d = a - b;
    return std::sqrt(d.dot(d));
}

Geometry GeometryFromState(const StateVec &x)
{
    const double r = std::clamp(x[8], 0.05, 0.60);
    const double l = std::clamp(x[9], -0.30, 0.30);
    const double side_radius = std::clamp(r + l, 0.05, 0.60);

    Geometry g;
    g.front_radius_m = r;
    g.back_radius_m = r;
    g.left_radius_m = side_radius;
    g.right_radius_m = side_radius;
    g.height_diff_m = std::clamp(x[10], -0.30, 0.30);
    return g;
}

void NormalizeState(StateVec &x)
{
    x[6] = WrapAngleRad(x[6]);
    x[7] = std::clamp(x[7], -8.0, 8.0);
    x[8] = std::clamp(x[8], 0.05, 0.60);
    x[9] = std::clamp(x[9], -0.30, 0.30);
    x[10] = std::clamp(x[10], -0.30, 0.30);
}

StateVec PredictState(const StateVec &x, double dt)
{
    StateVec y = x;
    y[0] += x[1] * dt;
    y[2] += x[3] * dt;
    y[4] += x[5] * dt;
    y[6] = WrapAngleRad(x[6] + x[7] * dt);
    NormalizeState(y);
    return y;
}

cv::Vec3d ArmorPositionFromState(const StateVec &x, ArmorSlot slot)
{
    const Geometry g = GeometryFromState(x);
    const cv::Vec3d center(x[0], x[2], x[4]);
    return center + RotateCameraY(SlotOffsetBody(slot, g), x[6]);
}

cv::Vec3d CameraYpd(const cv::Vec3d &point)
{
    const double horizontal = std::hypot(point[0], point[2]);
    const double distance = std::sqrt(point.dot(point));
    return cv::Vec3d(
        std::atan2(point[0], point[2]),
        std::atan2(-point[1], horizontal),
        distance);
}

MeasVec MeasureArmor(const StateVec &x, ArmorSlot slot)
{
    const cv::Vec3d armor = ArmorPositionFromState(x, slot);
    const cv::Vec3d ypd = CameraYpd(armor);
    return {ypd[0], ypd[1], ypd[2], WrapAngleRad(x[6] + SlotYawOffsetRad(slot))};
}

bool IsAngleMeasurementIndex(int idx)
{
    return idx == 0 || idx == 1 || idx == 3;
}

double AngleMean(const std::vector<double> &angles)
{
    double s = 0.0;
    double c = 0.0;
    for (double angle : angles)
    {
        s += std::sin(angle);
        c += std::cos(angle);
    }
    return std::atan2(s, c);
}

bool CholeskyLower(const StateMat &a, StateMat &l)
{
    for (auto &row : l)
    {
        row.fill(0.0);
    }

    for (int i = 0; i < kStateDim; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = a[i][j];
            for (int k = 0; k < j; ++k)
            {
                sum -= l[i][k] * l[j][k];
            }

            if (i == j)
            {
                if (sum <= 1e-12)
                {
                    return false;
                }
                l[i][j] = std::sqrt(sum);
            }
            else
            {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    return true;
}

class Ckf11
{
public:
    bool initialized() const { return initialized_; }
    const StateVec &state() const { return x_; }
    double lastNis() const { return last_stats_.nis; }
    double lastNees() const { return last_stats_.nees; }
    bool lastNisFail() const { return last_stats_.nis_fail; }
    bool lastNeesFail() const { return last_stats_.nees_fail; }
    double recentConsistencyFailureRate() const
    {
        if (consistency_window_size_ == 0)
        {
            return 0.0;
        }
        int failures = 0;
        for (std::size_t i = 0; i < consistency_window_size_; ++i)
        {
            failures += consistency_fail_window_[i];
        }
        return static_cast<double>(failures) / static_cast<double>(consistency_window_size_);
    }
    std::size_t consistencySampleCount() const { return consistency_window_size_; }

    void init(const StateVec &x0)
    {
        x_ = x0;
        NormalizeState(x_);

        for (auto &row : p_)
        {
            row.fill(0.0);
        }

        p_[0][0] = 1.0;
        p_[1][1] = 64.0;
        p_[2][2] = 1.0;
        p_[3][3] = 64.0;
        p_[4][4] = 1.0;
        p_[5][5] = 64.0;
        p_[6][6] = 0.4;
        p_[7][7] = 100.0;
        p_[8][8] = 1.0;
        p_[9][9] = 1.0;
        p_[10][10] = 1.0;
        last_stats_ = CkfUpdateStats{};
        consistency_fail_window_.fill(0);
        consistency_window_pos_ = 0;
        consistency_window_size_ = 0;
        initialized_ = true;
    }

    void predict(double dt)
    {
        if (!initialized_)
        {
            return;
        }

        const auto sigma = sigmaPoints();
        std::array<StateVec, 2 * kStateDim> propagated;
        for (std::size_t i = 0; i < propagated.size(); ++i)
        {
            propagated[i] = PredictState(sigma[i], dt);
        }

        x_.fill(0.0);
        std::vector<double> yaw_samples;
        yaw_samples.reserve(propagated.size());
        for (const StateVec &s : propagated)
        {
            for (int i = 0; i < kStateDim; ++i)
            {
                if (i != 6)
                {
                    x_[i] += s[i] / static_cast<double>(propagated.size());
                }
            }
            yaw_samples.push_back(s[6]);
        }
        x_[6] = AngleMean(yaw_samples);
        NormalizeState(x_);

        for (auto &row : p_)
        {
            row.fill(0.0);
        }
        for (const StateVec &s : propagated)
        {
            StateVec d{};
            for (int i = 0; i < kStateDim; ++i)
            {
                d[i] = (i == 6) ? WrapAngleRad(s[i] - x_[i]) : (s[i] - x_[i]);
            }
            for (int r = 0; r < kStateDim; ++r)
            {
                for (int c = 0; c < kStateDim; ++c)
                {
                    p_[r][c] += d[r] * d[c] / static_cast<double>(propagated.size());
                }
            }
        }
        addProcessNoise(dt);
    }

    CkfUpdateStats update(const MeasVec &z, ArmorSlot slot, double measurement_scale)
    {
        CkfUpdateStats stats;
        if (!initialized_)
        {
            last_stats_ = stats;
            return stats;
        }

        const StateVec x_prior = x_;
        const StateMat p_prior = p_;
        const auto sigma = sigmaPoints();
        std::array<MeasVec, 2 * kStateDim> z_sigma;
        for (std::size_t i = 0; i < z_sigma.size(); ++i)
        {
            z_sigma[i] = MeasureArmor(sigma[i], slot);
        }

        MeasVec z_mean{};
        std::array<std::vector<double>, kMeasDim> angle_samples;
        for (std::vector<double> &samples : angle_samples)
        {
            samples.reserve(z_sigma.size());
        }
        for (const MeasVec &s : z_sigma)
        {
            for (int i = 0; i < kMeasDim; ++i)
            {
                if (IsAngleMeasurementIndex(i))
                {
                    angle_samples[i].push_back(s[i]);
                }
                else
                {
                    z_mean[i] += s[i] / static_cast<double>(z_sigma.size());
                }
            }
        }
        for (int i = 0; i < kMeasDim; ++i)
        {
            if (IsAngleMeasurementIndex(i))
            {
                z_mean[i] = AngleMean(angle_samples[i]);
            }
        }

        MeasMat pzz{};
        CrossMat pxz{};
        for (std::size_t k = 0; k < sigma.size(); ++k)
        {
            StateVec dx{};
            MeasVec dz{};
            for (int i = 0; i < kStateDim; ++i)
            {
                dx[i] = (i == 6) ? WrapAngleRad(sigma[k][i] - x_[i]) : (sigma[k][i] - x_[i]);
            }
            for (int i = 0; i < kMeasDim; ++i)
            {
                dz[i] = IsAngleMeasurementIndex(i)
                            ? WrapAngleRad(z_sigma[k][i] - z_mean[i])
                            : (z_sigma[k][i] - z_mean[i]);
            }

            for (int r = 0; r < kMeasDim; ++r)
            {
                for (int c = 0; c < kMeasDim; ++c)
                {
                    pzz[r][c] += dz[r] * dz[c] / static_cast<double>(sigma.size());
                }
            }
            for (int r = 0; r < kStateDim; ++r)
            {
                for (int c = 0; c < kMeasDim; ++c)
                {
                    pxz[r][c] += dx[r] * dz[c] / static_cast<double>(sigma.size());
                }
            }
        }

        const double scale = std::clamp(measurement_scale, 1.0, 8.0);
        pzz[0][0] += scale * 4e-3;
        pzz[1][1] += scale * 4e-3;
        pzz[2][2] += scale * (1.0 + 0.08 * std::log(z[2] + 1.0));
        pzz[3][3] += scale * 9e-2;

        cv::Mat pzz_cv(kMeasDim, kMeasDim, CV_64F);
        for (int r = 0; r < kMeasDim; ++r)
        {
            for (int c = 0; c < kMeasDim; ++c)
            {
                pzz_cv.at<double>(r, c) = pzz[r][c];
            }
        }

        cv::Mat pzz_inv;
        if (!cv::invert(pzz_cv, pzz_inv, cv::DECOMP_CHOLESKY))
        {
            last_stats_ = stats;
            return stats;
        }

        MeasVec residual{};
        for (int i = 0; i < kMeasDim; ++i)
        {
            residual[i] = IsAngleMeasurementIndex(i)
                              ? WrapAngleRad(z[i] - z_mean[i])
                              : (z[i] - z_mean[i]);
        }

        stats.nis = 0.0;
        for (int i = 0; i < kMeasDim; ++i)
        {
            for (int j = 0; j < kMeasDim; ++j)
            {
                stats.nis += residual[i] * pzz_inv.at<double>(i, j) * residual[j];
            }
        }
        stats.nis_fail = std::isfinite(stats.nis) && stats.nis > kNisRejectThreshold;
        if (stats.nis_fail)
        {
            recordConsistencySample(true);
            last_stats_ = stats;
            return stats;
        }

        double k_gain[kStateDim][kMeasDim] = {};
        for (int r = 0; r < kStateDim; ++r)
        {
            for (int c = 0; c < kMeasDim; ++c)
            {
                for (int j = 0; j < kMeasDim; ++j)
                {
                    k_gain[r][c] += pxz[r][j] * pzz_inv.at<double>(j, c);
                }
            }
        }

        for (int r = 0; r < kStateDim; ++r)
        {
            for (int c = 0; c < kMeasDim; ++c)
            {
                x_[r] += k_gain[r][c] * residual[c];
            }
        }
        NormalizeState(x_);

        for (int r = 0; r < kStateDim; ++r)
        {
            for (int c = 0; c < kStateDim; ++c)
            {
                double reduction = 0.0;
                for (int i = 0; i < kMeasDim; ++i)
                {
                    for (int j = 0; j < kMeasDim; ++j)
                    {
                        reduction += k_gain[r][i] * pzz[i][j] * k_gain[c][j];
                    }
                }
                p_[r][c] -= reduction;
            }
        }
        stabilizeCovariance();

        StateVec correction{};
        for (int i = 0; i < kStateDim; ++i)
        {
            correction[i] = (i == 6) ? WrapAngleRad(x_[i] - x_prior[i]) : (x_[i] - x_prior[i]);
        }
        stats.nees = stateQuadraticForm(correction, p_prior);
        stats.nees_fail = std::isfinite(stats.nees) && stats.nees > kNeesRejectThreshold;
        if (stats.nees_fail)
        {
            x_ = x_prior;
            p_ = p_prior;
            recordConsistencySample(true);
            last_stats_ = stats;
            return stats;
        }

        stats.accepted = true;
        recordConsistencySample(false);
        last_stats_ = stats;
        return stats;
    }

private:
    double stateQuadraticForm(const StateVec &v, const StateMat &cov) const
    {
        cv::Mat cov_cv(kStateDim, kStateDim, CV_64F);
        for (double jitter : {1e-9, 1e-7, 1e-5, 1e-3})
        {
            for (int r = 0; r < kStateDim; ++r)
            {
                for (int c = 0; c < kStateDim; ++c)
                {
                    cov_cv.at<double>(r, c) = cov[r][c];
                }
                cov_cv.at<double>(r, r) += jitter;
            }

            cv::Mat cov_inv;
            if (!cv::invert(cov_cv, cov_inv, cv::DECOMP_CHOLESKY))
            {
                continue;
            }

            double q = 0.0;
            for (int i = 0; i < kStateDim; ++i)
            {
                for (int j = 0; j < kStateDim; ++j)
                {
                    q += v[i] * cov_inv.at<double>(i, j) * v[j];
                }
            }
            return q;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    void recordConsistencySample(bool failed)
    {
        consistency_fail_window_[consistency_window_pos_] = failed ? 1 : 0;
        consistency_window_pos_ = (consistency_window_pos_ + 1) % consistency_fail_window_.size();
        consistency_window_size_ = std::min<std::size_t>(consistency_window_size_ + 1, consistency_fail_window_.size());
    }

    std::array<StateVec, 2 * kStateDim> sigmaPoints()
    {
        StateMat cov = p_;
        StateMat l{};
        bool ok = false;
        for (double jitter : {1e-9, 1e-7, 1e-5, 1e-3})
        {
            for (int i = 0; i < kStateDim; ++i)
            {
                cov[i][i] = p_[i][i] + jitter;
            }
            if (CholeskyLower(cov, l))
            {
                ok = true;
                break;
            }
        }
        if (!ok)
        {
            init(x_);
            CholeskyLower(p_, l);
        }

        std::array<StateVec, 2 * kStateDim> sigma;
        const double scale = std::sqrt(static_cast<double>(kStateDim));
        for (int i = 0; i < kStateDim; ++i)
        {
            sigma[i] = x_;
            sigma[i + kStateDim] = x_;
            for (int r = 0; r < kStateDim; ++r)
            {
                sigma[i][r] += scale * l[r][i];
                sigma[i + kStateDim][r] -= scale * l[r][i];
            }
            NormalizeState(sigma[i]);
            NormalizeState(sigma[i + kStateDim]);
        }
        return sigma;
    }

    void addProcessNoise(double dt)
    {
        const double linear_acc_var = 100.0;
        const double angular_acc_var = 400.0;
        const double a = dt * dt * dt * dt / 4.0;
        const double b = dt * dt * dt / 2.0;
        const double c = dt * dt;

        for (const int pos : {0, 2, 4})
        {
            const int vel = pos + 1;
            p_[pos][pos] += a * linear_acc_var;
            p_[pos][vel] += b * linear_acc_var;
            p_[vel][pos] += b * linear_acc_var;
            p_[vel][vel] += c * linear_acc_var;
        }

        p_[6][6] += a * angular_acc_var;
        p_[6][7] += b * angular_acc_var;
        p_[7][6] += b * angular_acc_var;
        p_[7][7] += c * angular_acc_var;

        const double geom_var = 1e-5 * std::max(dt, 1e-3);
        p_[8][8] += geom_var;
        p_[9][9] += geom_var;
        p_[10][10] += geom_var;
    }

    void stabilizeCovariance()
    {
        for (int r = 0; r < kStateDim; ++r)
        {
            for (int c = r + 1; c < kStateDim; ++c)
            {
                const double v = 0.5 * (p_[r][c] + p_[c][r]);
                p_[r][c] = v;
                p_[c][r] = v;
            }
            p_[r][r] = std::max(p_[r][r], 1e-9);
        }

        StateMat l{};
        for (double jitter : {0.0, 1e-9, 1e-7, 1e-5, 1e-3})
        {
            StateMat test = p_;
            for (int i = 0; i < kStateDim; ++i)
            {
                test[i][i] += jitter;
            }
            if (CholeskyLower(test, l))
            {
                p_ = test;
                return;
            }
        }

        for (int r = 0; r < kStateDim; ++r)
        {
            for (int c = 0; c < kStateDim; ++c)
            {
                if (r != c)
                {
                    p_[r][c] = 0.0;
                }
            }
            p_[r][r] = std::max(p_[r][r], 1e-3);
        }
    }

    bool initialized_ = false;
    StateVec x_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.02, 0.0};
    StateMat p_{};
    CkfUpdateStats last_stats_;
    std::array<int, kConsistencyWindow> consistency_fail_window_{};
    std::size_t consistency_window_pos_ = 0;
    std::size_t consistency_window_size_ = 0;
};

MeasVec MeasurementFromPnp(const PnpResultMParam &pnp)
{
    const cv::Vec3d ypd = CameraYpd(pnp.tvec_m);
    return {ypd[0], ypd[1], ypd[2], ArmorYawRadFromPnp(pnp)};
}

MeasVec MeasurementFromPose(const PnpResultMParam::ArmorPose &pose)
{
    const cv::Vec3d ypd = CameraYpd(pose.tvec_m);
    return {ypd[0], ypd[1], ypd[2], ArmorYawRadFromPose(pose)};
}

MeasVec MeasurementResidual(const MeasVec &z, const MeasVec &pred)
{
    MeasVec residual{};
    for (int i = 0; i < kMeasDim; ++i)
    {
        residual[i] = IsAngleMeasurementIndex(i)
                          ? WrapAngleRad(z[i] - pred[i])
                          : (z[i] - pred[i]);
    }
    return residual;
}

double ArmorSlotMatchScore(const StateVec &x, ArmorSlot slot, const MeasVec &z)
{
    const MeasVec pred = MeasureArmor(x, slot);
    const MeasVec residual = MeasurementResidual(z, pred);
    return std::abs(residual[3]) + std::abs(residual[0]);
}

ArmorSlot BestArmorSlot(const StateVec &x, const MeasVec &z)
{
    const std::array<ArmorSlot, 4> slots = {
        ArmorSlot::FRONT,
        ArmorSlot::LEFT,
        ArmorSlot::BACK,
        ArmorSlot::RIGHT};

    std::vector<std::pair<ArmorSlot, MeasVec>> candidates;
    candidates.reserve(slots.size());
    for (ArmorSlot slot : slots)
    {
        candidates.push_back({slot, MeasureArmor(x, slot)});
    }
    std::sort(candidates.begin(),
              candidates.end(),
              [](const auto &a, const auto &b) {
                  return a.second[2] < b.second[2];
              });

    ArmorSlot best = ArmorSlot::FRONT;
    double best_error = std::numeric_limits<double>::max();
    const std::size_t keep_count = std::min<std::size_t>(3, candidates.size());
    for (std::size_t i = 0; i < keep_count; ++i)
    {
        const ArmorSlot slot = candidates[i].first;
        const double score = ArmorSlotMatchScore(x, slot, z);
        if (score < best_error)
        {
            best_error = score;
            best = slot;
        }
    }
    return best;
}

bool PassMeasurementGate(const MeasVec &z, const MeasVec &pred)
{
    const MeasVec residual = MeasurementResidual(z, pred);
    const double distance_gate = std::clamp(0.45 + 0.18 * z[2], 0.60, 1.80);
    return std::abs(residual[0]) < 0.75 &&
           std::abs(residual[1]) < 0.50 &&
           std::abs(residual[2]) < distance_gate &&
           std::abs(residual[3]) < 1.20;
}

StateVec InitialStateFromPnp(const PnpResultMParam &pnp, ArmorSlot slot, const Geometry &geometry)
{
    const double armor_yaw = ArmorYawRadFromPnp(pnp);
    const double body_yaw = WrapAngleRad(armor_yaw - SlotYawOffsetRad(slot));
    const cv::Vec3d center = pnp.tvec_m - RotateCameraY(SlotOffsetBody(slot, geometry), body_yaw);

    return {center[0], 0.0,
            center[1], 0.0,
            center[2], 0.0,
            body_yaw, 0.0,
            geometry.front_radius_m,
            geometry.left_radius_m - geometry.front_radius_m,
            geometry.height_diff_m};
}

StateVec InitialStateFromPose(const PnpResultMParam::ArmorPose &pose, ArmorSlot slot, const Geometry &geometry)
{
    const double armor_yaw = ArmorYawRadFromPose(pose);
    const double body_yaw = WrapAngleRad(armor_yaw - SlotYawOffsetRad(slot));
    const cv::Vec3d center = pose.tvec_m - RotateCameraY(SlotOffsetBody(slot, geometry), body_yaw);

    return {center[0], 0.0,
            center[1], 0.0,
            center[2], 0.0,
            body_yaw, 0.0,
            geometry.front_radius_m,
            geometry.left_radius_m - geometry.front_radius_m,
            geometry.height_diff_m};
}

ArmorSlot SelectInitialSlotFromPnp(const PnpResultMParam &pnp, const Geometry &geometry)
{
    const std::array<ArmorSlot, 4> slots = {
        ArmorSlot::FRONT,
        ArmorSlot::LEFT,
        ArmorSlot::BACK,
        ArmorSlot::RIGHT};

    const cv::Vec3d armor_ypd = CameraYpd(pnp.tvec_m);
    ArmorSlot best = ArmorSlot::FRONT;
    double best_score = std::numeric_limits<double>::max();
    for (ArmorSlot slot : slots)
    {
        const StateVec candidate = InitialStateFromPnp(pnp, slot, geometry);
        const cv::Vec3d center(candidate[0], candidate[2], candidate[4]);
        const cv::Vec3d center_ypd = CameraYpd(center);
        double score = std::abs(WrapAngleRad(center_ypd[0] - armor_ypd[0])) +
                       0.4 * std::abs(center_ypd[1] - armor_ypd[1]) +
                       0.05 * std::abs(center_ypd[2] - armor_ypd[2]);
        if (center[2] <= 0.05)
        {
            score += 1000.0;
        }
        if (score < best_score)
        {
            best_score = score;
            best = slot;
        }
    }
    return best;
}

ArmorSlot SelectInitialSlotFromPose(const PnpResultMParam::ArmorPose &pose, const Geometry &geometry)
{
    const std::array<ArmorSlot, 4> slots = {
        ArmorSlot::FRONT,
        ArmorSlot::LEFT,
        ArmorSlot::BACK,
        ArmorSlot::RIGHT};

    const cv::Vec3d armor_ypd = CameraYpd(pose.tvec_m);
    ArmorSlot best = ArmorSlot::FRONT;
    double best_score = std::numeric_limits<double>::max();
    for (ArmorSlot slot : slots)
    {
        const StateVec candidate = InitialStateFromPose(pose, slot, geometry);
        const cv::Vec3d center(candidate[0], candidate[2], candidate[4]);
        const cv::Vec3d center_ypd = CameraYpd(center);
        double score = std::abs(WrapAngleRad(center_ypd[0] - armor_ypd[0])) +
                       0.4 * std::abs(center_ypd[1] - armor_ypd[1]) +
                       0.05 * std::abs(center_ypd[2] - armor_ypd[2]);
        if (center[2] <= 0.05)
        {
            score += 1000.0;
        }
        if (score < best_score)
        {
            best_score = score;
            best = slot;
        }
    }
    return best;
}

std::array<ArmorPose, 4> GenerateArmors(const StateVec &x)
{
    const Geometry g = GeometryFromState(x);
    const cv::Vec3d center(x[0], x[2], x[4]);
    const std::array<ArmorSlot, 4> slots = {
        ArmorSlot::FRONT,
        ArmorSlot::LEFT,
        ArmorSlot::BACK,
        ArmorSlot::RIGHT};

    std::array<ArmorPose, 4> armors;
    for (std::size_t i = 0; i < slots.size(); ++i)
    {
        const ArmorSlot slot = slots[i];
        armors[i].slot = slot;
        armors[i].center_m = ArmorPositionFromState(x, slot);
        armors[i].yaw_rad = WrapAngleRad(x[6] + SlotYawOffsetRad(slot));
        armors[i].pitch_rad = kArmorTiltRad;
    }
    return armors;
}

void WriteJsonNumber(std::ofstream &fout, double value)
{
    if (std::isfinite(value))
    {
        fout << value;
    }
    else
    {
        fout << "null";
    }
}

void DumpCkfForFoxglove(uint64_t frame_id,
                        int infer_id,
                        bool has_state,
                        ArmorSlot selected_slot,
                        const std::string &target_armor_name,
                        double nis,
                        double nees,
                        bool nis_fail,
                        bool nees_fail,
                        double consistency_failure_rate,
                        const StateVec &x,
                        const std::array<ArmorPose, 4> &armors)
{
    if (!IsFoxgloveDebugEnabled())
    {
        return;
    }

    static const std::string kDir = "/tmp/rm_rerun";
    static const std::string kTmp = kDir + "/rgo_latest.tmp.json";
    static const std::string kOut = kDir + "/rgo_latest.json";
    static bool init = false;
    if (!init)
    {
        std::error_code ec;
        std::filesystem::create_directories(kDir, ec);
        init = true;
    }

    std::ofstream fout(kTmp, std::ios::trunc);
    if (!fout.good())
    {
        return;
    }

    fout << "{\n";
    fout << "  \"tracker\": \"ckf11\",\n";
    fout << "  \"frame_id\": " << frame_id << ",\n";
    fout << "  \"infer_id\": " << infer_id << ",\n";
    fout << "  \"has_state\": " << (has_state ? "true" : "false") << ",\n";
    fout << "  \"selected_slot\": \"" << SlotName(selected_slot) << "\",\n";
    fout << "  \"target_armor_name\": \"" << target_armor_name << "\",\n";
    fout << "  \"nis\": ";
    WriteJsonNumber(fout, nis);
    fout << ",\n";
    fout << "  \"nees\": ";
    WriteJsonNumber(fout, nees);
    fout << ",\n";
    fout << "  \"nis_fail\": " << (nis_fail ? "true" : "false") << ",\n";
    fout << "  \"nees_fail\": " << (nees_fail ? "true" : "false") << ",\n";
    fout << "  \"consistency_failure_rate\": " << consistency_failure_rate << ",\n";
    fout << "  \"body_center_m\": [" << x[0] << ", " << x[2] << ", " << x[4] << "],\n";
    fout << "  \"body_yaw_rad\": " << x[6] << ",\n";
    fout << "  \"body_velocity_mps\": [" << x[1] << ", " << x[3] << ", " << x[5] << "],\n";
    fout << "  \"body_yaw_rate_rad_s\": " << x[7] << ",\n";
    fout << "  \"state\": [" << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << ", "
         << x[4] << ", " << x[5] << ", " << x[6] << ", " << x[7] << ", "
         << x[8] << ", " << x[9] << ", " << x[10] << "],\n";
    fout << "  \"armors\": [\n";
    for (std::size_t i = 0; i < armors.size(); ++i)
    {
        const ArmorPose &armor = armors[i];
        fout << "    {\"slot\": \"" << SlotName(armor.slot) << "\", "
             << "\"center_m\": [" << armor.center_m[0] << ", " << armor.center_m[1] << ", " << armor.center_m[2] << "], "
             << "\"yaw_rad\": " << armor.yaw_rad << ", "
             << "\"pitch_rad\": " << armor.pitch_rad << "}";
        fout << (i + 1 < armors.size() ? ",\n" : "\n");
    }
    fout << "  ]\n";
    fout << "}\n";
    fout.close();
    std::rename(kTmp.c_str(), kOut.c_str());
}

class CkfTrackerNode : public GNode
{
public:
    CStatus init() override
    {
        pnp_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(PnpResultMParam, PNP_TOPIC, 2);
        rgo_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(RgoOutputMParam, RGO_OUTPUT_TOPIC, 2);
        const int pool_size = GetAppConfig().rgo_message_pool_size > 0
                                  ? GetAppConfig().rgo_message_pool_size
                                  : 4;
        output_pool_.preallocate(static_cast<std::size_t>(pool_size));
        return CStatus();
    }

    CStatus run() override
    {
        const AppConfig &cfg = GetAppConfig();
        app::runtime::ApplyThreadAffinity("ckf_tracker", cfg.affinity_enable ? cfg.rgo_cpu : -1);

        while (!IsYoloStopRequested())
        {
            std::shared_ptr<PnpResultMParam> pnp = nullptr;
            const CStatus st = CGRAPH_SUB_MPARAM_WITH_TIMEOUT(PnpResultMParam, pnp_conn_id_, pnp, 1000);
            if (st.isErr() || !pnp)
            {
                continue;
            }

            std::shared_ptr<RgoOutputMParam> output = output_pool_.acquire();
            output->frame_id = pnp->frame_id;
            output->infer_id = pnp->infer_id;
            output->latency_ms = pnp->latency_ms;
            output->detect_latency_ms = pnp->detect_latency_ms;
            output->status = pnp->status;
            output->armor_type = pnp->armor_type;
            output->class_id = pnp->class_id;
            output->armor_name_id = pnp->armor_name_id;
            output->armor_name_confidence = pnp->armor_name_confidence;
            output->armor_name = pnp->armor_name;
            output->mean_reprojection_error_px = pnp->mean_reprojection_error_px;
            output->max_reprojection_error_px = pnp->max_reprojection_error_px;
            output->reprojection_ok = pnp->reprojection_ok;
            output->depth_ok = pnp->depth_ok;

            const auto now = std::chrono::steady_clock::now();
            double dt = 1.0 / 120.0;
            if (has_update_tp_)
            {
                dt = std::chrono::duration<double>(now - last_update_tp_).count();
                dt = std::clamp(dt, 1.0 / 300.0, 0.2);
            }
            last_update_tp_ = now;
            has_update_tp_ = true;

            if (ckf_.initialized())
            {
                ckf_.predict(dt);
            }

            if (pnp->has_pose)
            {
                std::vector<PnpResultMParam::ArmorPose> armor_observations = pnp->armors;
                if (armor_observations.empty())
                {
                    PnpResultMParam::ArmorPose pose;
                    pose.has_pose = pnp->has_pose;
                    pose.armor_type = pnp->armor_type;
                    pose.class_id = pnp->class_id;
                    pose.armor_name_id = pnp->armor_name_id;
                    pose.armor_name_confidence = pnp->armor_name_confidence;
                    pose.armor_name = pnp->armor_name;
                    pose.mean_reprojection_error_px = pnp->mean_reprojection_error_px;
                    pose.max_reprojection_error_px = pnp->max_reprojection_error_px;
                    pose.reprojection_ok = pnp->reprojection_ok;
                    pose.depth_ok = pnp->depth_ok;
                    pose.center_px = pnp->center_px;
                    pose.tvec_m = pnp->tvec_m;
                    pose.rvec = pnp->rvec;
                    pose.ypr_deg = pnp->ypr_deg;
                    pose.quat_xyzw = pnp->quat_xyzw;
                    pose.armor_size_m = pnp->armor_size_m;
                    armor_observations.push_back(std::move(pose));
                }

                bool updated = false;
                bool saw_different_named_target = false;
                PnpResultMParam::ArmorPose switch_candidate;
                for (const PnpResultMParam::ArmorPose &pose : armor_observations)
                {
                    if (!pose.has_pose)
                    {
                        continue;
                    }
                    const bool has_name = pose.armor_name_id >= 0 && !pose.armor_name.empty();
                    if (!ckf_.initialized())
                    {
                        selected_slot_ = SelectInitialSlotFromPose(pose, default_geometry_);
                        target_armor_name_id_ = has_name ? pose.armor_name_id : -1;
                        target_armor_name_ = has_name ? pose.armor_name : std::string();
                        name_mismatch_count_ = 0;
                        gate_fail_count_ = 0;
                        ckf_.init(InitialStateFromPose(pose, selected_slot_, default_geometry_));
                        updated = true;
                        continue;
                    }

                    if (target_armor_name_id_ >= 0 && has_name && pose.armor_name_id != target_armor_name_id_)
                    {
                        saw_different_named_target = true;
                        switch_candidate = pose;
                        continue;
                    }

                    if (target_armor_name_id_ < 0 && has_name)
                    {
                        target_armor_name_id_ = pose.armor_name_id;
                        target_armor_name_ = pose.armor_name;
                    }

                    const MeasVec z = MeasurementFromPose(pose);
                    selected_slot_ = BestArmorSlot(ckf_.state(), z);
                    const MeasVec predicted_z = MeasureArmor(ckf_.state(), selected_slot_);
                    if (!PassMeasurementGate(z, predicted_z))
                    {
                        continue;
                    }

                    name_mismatch_count_ = 0;
                    const double measurement_scale =
                        (pose.reprojection_ok && pose.depth_ok) ? 1.0 : 4.0;
                    const CkfUpdateStats stats = ckf_.update(z, selected_slot_, measurement_scale);
                    if (!stats.accepted)
                    {
                        spdlog::debug("[CKF] reject update frame={} infer={} slot={} nis={} nees={} nis_fail={} nees_fail={}",
                                      pnp->frame_id,
                                      pnp->infer_id,
                                      SlotName(selected_slot_),
                                      stats.nis,
                                      stats.nees,
                                      stats.nis_fail,
                                      stats.nees_fail);
                        continue;
                    }

                    gate_fail_count_ = 0;
                    updated = true;
                }

                if (!updated && ckf_.initialized())
                {
                    if (saw_different_named_target)
                    {
                        ++name_mismatch_count_;
                        if (name_mismatch_count_ >= 30)
                        {
                            selected_slot_ = SelectInitialSlotFromPose(switch_candidate, default_geometry_);
                            target_armor_name_id_ = switch_candidate.armor_name_id;
                            target_armor_name_ = switch_candidate.armor_name;
                            name_mismatch_count_ = 0;
                            gate_fail_count_ = 0;
                            ckf_.init(InitialStateFromPose(switch_candidate, selected_slot_, default_geometry_));
                        }
                    }
                    else
                    {
                        ++gate_fail_count_;
                        if (gate_fail_count_ >= 3 && !armor_observations.empty())
                        {
                            selected_slot_ = SelectInitialSlotFromPose(armor_observations.front(), default_geometry_);
                            gate_fail_count_ = 0;
                            ckf_.init(InitialStateFromPose(armor_observations.front(), selected_slot_, default_geometry_));
                        }
                    }
                }
            }

            output->has_target = ckf_.initialized();
            if (ckf_.initialized())
            {
                const StateVec &x = ckf_.state();
                output->xyz_m = cv::Vec3d(x[0], x[2], x[4]);
                output->yaw_deg = x[6] * kRadToDeg;
                output->bearing_yaw_deg = std::atan2(x[0], x[4]) * kRadToDeg;
                DumpCkfForFoxglove(pnp->frame_id,
                                    pnp->infer_id,
                                    true,
                                    selected_slot_,
                                    target_armor_name_,
                                    ckf_.lastNis(),
                                    ckf_.lastNees(),
                                    ckf_.lastNisFail(),
                                    ckf_.lastNeesFail(),
                                    ckf_.recentConsistencyFailureRate(),
                                    x,
                                    GenerateArmors(x));
            }
            else
            {
                output->xyz_m = cv::Vec3d(0.0, 0.0, 0.0);
                output->yaw_deg = 0.0;
                output->bearing_yaw_deg = 0.0;
                DumpCkfForFoxglove(pnp->frame_id,
                                    pnp->infer_id,
                                    false,
                                    selected_slot_,
                                    target_armor_name_,
                                    ckf_.lastNis(),
                                    ckf_.lastNees(),
                                    ckf_.lastNisFail(),
                                    ckf_.lastNeesFail(),
                                    ckf_.recentConsistencyFailureRate(),
                                    empty_state_,
                                    GenerateArmors(empty_state_));
            }

            const CStatus pub_st = CGRAPH_PUB_MPARAM(RgoOutputMParam, RGO_OUTPUT_TOPIC, output, GMessagePushStrategy::REPLACE);
            if (pub_st.isErr())
            {
                spdlog::warn("[CKF] publish output failed frame={} infer={} error={}",
                             pnp->frame_id,
                             pnp->infer_id,
                             pub_st.getInfo());
            }
        }
        return CStatus();
    }

private:
    int pnp_conn_id_ = -1;
    int rgo_conn_id_ = -1;
    SharedParamPool<RgoOutputMParam> output_pool_;

    Ckf11 ckf_;
    Geometry default_geometry_;
    ArmorSlot selected_slot_ = ArmorSlot::FRONT;
    int gate_fail_count_ = 0;
    int target_armor_name_id_ = -1;
    std::string target_armor_name_;
    int name_mismatch_count_ = 0;
    bool has_update_tp_ = false;
    std::chrono::steady_clock::time_point last_update_tp_;
    StateVec empty_state_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.02, 0.0};
};

} // namespace

void RegisterRgoPipelineElements(CGraph::GPipeline *const &pipeline,
                                 CGraph::GElementPtr *rgo_ref,
                                 const CGraph::GElementPtrSet &depends)
{
    GElementPtr ckf = nullptr;
    CStatus st = pipeline->registerGElement<CkfTrackerNode>(&ckf, depends, "CKF 跟踪\nckf_tracker");
    if (st.isErr())
    {
        spdlog::error("register ckf tracker failed: {}", st.getInfo());
    }
    if (rgo_ref)
    {
        *rgo_ref = ckf;
    }
}
