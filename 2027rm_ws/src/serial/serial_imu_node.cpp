#include <array>
#include <cerrno>
#include <chrono>
#include <coroutine>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <sys/epoll.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

#ifdef CSIZE
constexpr tcflag_t kTermiosCsizeMask = CSIZE;
#else
constexpr tcflag_t kTermiosCsizeMask = CS5 | CS6 | CS7 | CS8;
#endif

#ifdef CSIZE
#undef CSIZE
#endif
#ifdef CSTATUS
#undef CSTATUS
#endif

#include <CGraph.h>
#include <spdlog/spdlog.h>

#include "message_pool.hpp"
#include "serial_protocol.hpp"
#include "thread_affinity.hpp"
#include "yolo_app.hpp"
#include "yolo_common.hpp"

using namespace CGraph;

namespace {

struct SerialFrame
{
    app::serial::HeaderFrame header{};
    std::array<uint8_t, app::serial::kHeaderSize + app::serial::kMaxPayloadLen + app::serial::kCrc16Size> bytes{};
    std::size_t size = 0;
};

speed_t ToSpeed(int baud_rate)
{
    switch (baud_rate)
    {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return B460800;
        case 500000: return B500000;
        case 576000: return B576000;
        case 921600: return B921600;
#ifdef B1000000
        case 1000000: return B1000000;
#endif
#ifdef B1500000
        case 1500000: return B1500000;
#endif
        default:
            throw std::invalid_argument("unsupported baud_rate: " + std::to_string(baud_rate));
    }
}

class SerialPort
{
public:
    ~SerialPort()
    {
        close();
    }

    void open(const std::string &device, int baud_rate)
    {
        close();
        fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC | O_NONBLOCK);
        if (fd_ < 0)
        {
            throw std::system_error(errno, std::generic_category(), "open " + device);
        }

        termios tio{};
        if (tcgetattr(fd_, &tio) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "tcgetattr " + device);
        }

        cfmakeraw(&tio);
        const speed_t speed = ToSpeed(baud_rate);
        cfsetispeed(&tio, speed);
        cfsetospeed(&tio, speed);
        tio.c_cflag |= static_cast<tcflag_t>(CLOCAL | CREAD);
        tio.c_cflag &= static_cast<tcflag_t>(~CRTSCTS);
        tio.c_cflag &= static_cast<tcflag_t>(~CSTOPB);
        tio.c_cflag &= static_cast<tcflag_t>(~PARENB);
        tio.c_cflag &= static_cast<tcflag_t>(~kTermiosCsizeMask);
        tio.c_cflag |= CS8;
        tio.c_cc[VMIN] = 0;
        tio.c_cc[VTIME] = 0;

        if (tcsetattr(fd_, TCSANOW, &tio) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "tcsetattr " + device);
        }
        tcflush(fd_, TCIOFLUSH);
    }

    void close()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ssize_t readSome(uint8_t *dst, std::size_t len)
    {
        return ::read(fd_, dst, len);
    }

    int fd() const
    {
        return fd_;
    }

private:
    int fd_ = -1;
};

class IoAwaiter
{
public:
    virtual ~IoAwaiter() = default;
    virtual void onReady() = 0;
};

class EpollLoop
{
public:
    EpollLoop()
    {
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0)
        {
            throw std::system_error(errno, std::generic_category(), "epoll_create1");
        }
    }

    ~EpollLoop()
    {
        if (epoll_fd_ >= 0)
        {
            ::close(epoll_fd_);
        }
    }

    void watchReadable(int fd, IoAwaiter *awaiter)
    {
        pending_ = awaiter;
        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLERR | EPOLLHUP;
        ev.data.fd = fd;
        const int op = registered_ ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
        if (epoll_ctl(epoll_fd_, op, fd, &ev) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "epoll_ctl");
        }
        registered_ = true;
    }

    void clear(IoAwaiter *awaiter)
    {
        if (pending_ == awaiter)
        {
            pending_ = nullptr;
        }
    }

    void pollOnce(int timeout_ms)
    {
        epoll_event ev{};
        const int n = epoll_wait(epoll_fd_, &ev, 1, timeout_ms);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                return;
            }
            throw std::system_error(errno, std::generic_category(), "epoll_wait");
        }
        if (n > 0 && pending_)
        {
            pending_->onReady();
        }
    }

private:
    int epoll_fd_ = -1;
    bool registered_ = false;
    IoAwaiter *pending_ = nullptr;
};

class ReadExactAwaiter : public IoAwaiter
{
public:
    ReadExactAwaiter(SerialPort &port, EpollLoop &loop, uint8_t *dst, std::size_t len)
        : port_(port), loop_(loop), dst_(dst), len_(len)
    {
    }

    bool await_ready()
    {
        return tryRead();
    }

    void await_suspend(std::coroutine_handle<> handle)
    {
        handle_ = handle;
        loop_.watchReadable(port_.fd(), this);
    }

    bool await_resume()
    {
        if (error_)
        {
            std::rethrow_exception(error_);
        }
        return offset_ == len_;
    }

    void onReady() override
    {
        if (tryRead())
        {
            loop_.clear(this);
            handle_.resume();
        }
    }

private:
    bool tryRead()
    {
        while (offset_ < len_ && !IsYoloStopRequested())
        {
            const ssize_t n = port_.readSome(dst_ + offset_, len_ - offset_);
            if (n > 0)
            {
                offset_ += static_cast<std::size_t>(n);
                continue;
            }
            if (n == 0 || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                return false;
            }
            if (errno == EINTR)
            {
                continue;
            }
            error_ = std::make_exception_ptr(
                std::system_error(errno, std::generic_category(), "serial read"));
            return true;
        }
        return offset_ == len_ || IsYoloStopRequested();
    }

    SerialPort &port_;
    EpollLoop &loop_;
    uint8_t *dst_ = nullptr;
    std::size_t len_ = 0;
    std::size_t offset_ = 0;
    std::coroutine_handle<> handle_{};
    std::exception_ptr error_;
};

class Task
{
public:
    struct promise_type
    {
        Task get_return_object()
        {
            return Task(std::coroutine_handle<promise_type>::from_promise(*this));
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() { exception = std::current_exception(); }

        std::exception_ptr exception;
    };

    explicit Task(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
    Task(const Task &) = delete;
    Task &operator=(const Task &) = delete;
    Task(Task &&other) noexcept : handle_(other.handle_) { other.handle_ = {}; }
    ~Task()
    {
        if (handle_) handle_.destroy();
    }

    void start()
    {
        if (handle_)
        {
            handle_.resume();
        }
    }

    bool done() const
    {
        return !handle_ || handle_.done();
    }

    void rethrowIfFailed()
    {
        if (handle_ && handle_.promise().exception)
        {
            std::rethrow_exception(handle_.promise().exception);
        }
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

ReadExactAwaiter ReadExact(SerialPort &port, EpollLoop &loop, uint8_t *dst, std::size_t len)
{
    return ReadExactAwaiter(port, loop, dst, len);
}

template <typename Callback>
Task ReadFramesAsync(SerialPort &port, EpollLoop &loop, Callback on_frame)
{
    while (!IsYoloStopRequested())
    {
        uint8_t sof = 0;
        if (!co_await ReadExact(port, loop, &sof, 1))
        {
            co_return;
        }
        if (sof != app::serial::kSof)
        {
            continue;
        }

        SerialFrame frame;
        frame.bytes[0] = sof;
        if (!co_await ReadExact(port, loop, frame.bytes.data() + 1, app::serial::kHeaderSize - 1))
        {
            co_return;
        }

        std::memcpy(&frame.header, frame.bytes.data(), sizeof(app::serial::HeaderFrame));
        if (!app::serial::VerifyCrc8(frame.bytes.data(), app::serial::kHeaderSize))
        {
            continue;
        }

        frame.size = app::serial::kHeaderSize + frame.header.len + app::serial::kCrc16Size;
        if (frame.size > frame.bytes.size())
        {
            continue;
        }
        if (!co_await ReadExact(port, loop, frame.bytes.data() + app::serial::kHeaderSize, frame.header.len + app::serial::kCrc16Size))
        {
            co_return;
        }
        if (!app::serial::VerifyCrc16(frame.bytes.data(), frame.size))
        {
            continue;
        }

        on_frame(frame);
    }
}

bool DecodeImuFrame(const SerialFrame &frame, app::serial::ReceiveImuData &imu)
{
    if (frame.header.id != app::serial::kIdImu || frame.size != sizeof(app::serial::ReceiveImuData))
    {
        return false;
    }
    std::memcpy(&imu, frame.bytes.data(), sizeof(app::serial::ReceiveImuData));
    return true;
}

class SerialImuNode : public GNode
{
public:
    CStatus init() override
    {
        imu_conn_id_ = CGRAPH_BIND_MESSAGE_TOPIC(ImuMParam, IMU_TOPIC, 2);
        const int pool_size = GetAppConfig().imu_message_pool_size > 0
                                  ? GetAppConfig().imu_message_pool_size
                                  : 4;
        imu_pool_.preallocate(static_cast<std::size_t>(pool_size));
        return CStatus();
    }

    CStatus run() override
    {
        const AppConfig &cfg = GetAppConfig();
        app::runtime::ApplyThreadAffinity("serial", cfg.affinity_enable ? cfg.serial_cpu : -1);
        if (!cfg.serial_enable)
        {
            spdlog::info("[SERIAL] disabled");
            return CStatus();
        }

        uint64_t sequence = 0;
        while (!IsYoloStopRequested())
        {
            try
            {
                port_.open(cfg.serial_device, cfg.serial_baud_rate);
                spdlog::info("[SERIAL] opened {} baud={}", cfg.serial_device, cfg.serial_baud_rate);
                EpollLoop loop;
                auto task = ReadFramesAsync(port_, loop, [this, &sequence](const SerialFrame &frame) {
                    app::serial::ReceiveImuData imu{};
                    if (!DecodeImuFrame(frame, imu))
                    {
                        return;
                    }
                    std::shared_ptr<ImuMParam> out = imu_pool_.acquire();
                    out->time_stamp_ms = imu.time_stamp;
                    out->sequence = ++sequence;
                    out->receive_tp = std::chrono::steady_clock::now();
                    out->yaw_rad = imu.data.yaw;
                    out->pitch_rad = imu.data.pitch;
                    out->roll_rad = imu.data.roll;
                    out->yaw_vel_rad_s = imu.data.yaw_vel;
                    out->pitch_vel_rad_s = imu.data.pitch_vel;
                    out->roll_vel_rad_s = imu.data.roll_vel;
                    CStatus st = CGRAPH_PUB_MPARAM(ImuMParam, IMU_TOPIC, out, GMessagePushStrategy::REPLACE);
                    if (st.isErr())
                    {
                        spdlog::warn("[SERIAL] publish imu failed: {}", st.getInfo());
                    }
                });
                task.start();
                while (!IsYoloStopRequested() && !task.done())
                {
                    loop.pollOnce(100);
                    task.rethrowIfFailed();
                }
                task.rethrowIfFailed();
            }
            catch (const std::exception &e)
            {
                spdlog::warn("[SERIAL] {}", e.what());
                port_.close();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        port_.close();
        return CStatus();
    }

    CStatus destroy() override
    {
        port_.close();
        return CStatus();
    }

private:
    int imu_conn_id_ = -1;
    SerialPort port_;
    SharedParamPool<ImuMParam> imu_pool_;
};

}  // namespace

void RegisterSerialImuPipelineElements(CGraph::GPipeline* const &pipeline,
                                       CGraph::GElementPtr *serial_ref,
                                       const CGraph::GElementPtrSet &depends)
{
    GElementPtr serial = nullptr;
    CStatus st = pipeline->registerGElement<SerialImuNode>(&serial, depends, "串口IMU\nserial_imu");
    if (st.isErr())
    {
        spdlog::error("register serial imu failed: {}", st.getInfo());
    }
    if (serial_ref)
    {
        *serial_ref = serial;
    }
}
