#pragma once

#include <cstddef>
#include <cstdint>

namespace app::serial
{

inline constexpr uint8_t kSof = 0x5A;
inline constexpr uint8_t kIdDebug = 0x01;
inline constexpr uint8_t kIdImu = 0x02;
inline constexpr uint8_t kIdRobotStateInfo = 0x03;
inline constexpr uint8_t kIdEventData = 0x04;
inline constexpr uint8_t kIdPidDebug = 0x05;
inline constexpr uint8_t kIdAllRobotHp = 0x06;
inline constexpr uint8_t kIdGameStatus = 0x07;
inline constexpr uint8_t kIdRobotMotion = 0x08;
inline constexpr uint8_t kIdGroundRobotPosition = 0x09;
inline constexpr uint8_t kIdRfidStatus = 0x0A;
inline constexpr uint8_t kIdRobotStatus = 0x0B;
inline constexpr uint8_t kIdJointState = 0x0C;
inline constexpr uint8_t kIdBuff = 0x0D;
inline constexpr uint8_t kIdRobotCmd = 0x01;

inline constexpr std::size_t kHeaderSize = 4;
inline constexpr std::size_t kCrc16Size = 2;
inline constexpr std::size_t kMaxPayloadLen = 255;

struct HeaderFrame
{
    uint8_t sof;
    uint8_t len;
    uint8_t id;
    uint8_t crc;
} __attribute__((packed));

struct ReceiveImuData
{
    HeaderFrame frame_header;
    uint32_t time_stamp;
    struct
    {
        float yaw;
        float pitch;
        float roll;
        float yaw_vel;
        float pitch_vel;
        float roll_vel;
    } __attribute__((packed)) data;
    uint16_t crc;
} __attribute__((packed));

uint8_t GetCrc8(const uint8_t *data, std::size_t len);
bool VerifyCrc8(const uint8_t *data, std::size_t len);
void AppendCrc8(uint8_t *data, std::size_t len);

uint16_t GetCrc16(const uint8_t *data, std::size_t len);
bool VerifyCrc16(const uint8_t *data, std::size_t len);
void AppendCrc16(uint8_t *data, std::size_t len);

}  // namespace app::serial
