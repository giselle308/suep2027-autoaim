#include "serial_protocol.hpp"

namespace app::serial
{

uint8_t GetCrc8(const uint8_t *data, std::size_t len)
{
    uint8_t crc = 0xff;
    for (std::size_t i = 0; i < len; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            crc = (crc & 0x01) ? static_cast<uint8_t>((crc >> 1) ^ 0x8c)
                               : static_cast<uint8_t>(crc >> 1);
        }
    }
    return crc;
}

bool VerifyCrc8(const uint8_t *data, std::size_t len)
{
    if (!data || len <= 2)
    {
        return false;
    }
    return GetCrc8(data, len - 1) == data[len - 1];
}

void AppendCrc8(uint8_t *data, std::size_t len)
{
    if (!data || len <= 2)
    {
        return;
    }
    data[len - 1] = GetCrc8(data, len - 1);
}

uint16_t GetCrc16(const uint8_t *data, std::size_t len)
{
    uint16_t crc = 0xffff;
    for (std::size_t i = 0; i < len; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            crc = (crc & 0x0001) ? static_cast<uint16_t>((crc >> 1) ^ 0x8408)
                                 : static_cast<uint16_t>(crc >> 1);
        }
    }
    return crc;
}

bool VerifyCrc16(const uint8_t *data, std::size_t len)
{
    if (!data || len <= 2)
    {
        return false;
    }
    const uint16_t expected = GetCrc16(data, len - 2);
    return static_cast<uint8_t>(expected & 0x00ff) == data[len - 2] &&
           static_cast<uint8_t>((expected >> 8) & 0x00ff) == data[len - 1];
}

void AppendCrc16(uint8_t *data, std::size_t len)
{
    if (!data || len <= 2)
    {
        return;
    }
    const uint16_t crc = GetCrc16(data, len - 2);
    data[len - 2] = static_cast<uint8_t>(crc & 0x00ff);
    data[len - 1] = static_cast<uint8_t>((crc >> 8) & 0x00ff);
}

}  // namespace app::serial
