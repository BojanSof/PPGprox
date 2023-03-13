#ifndef _PPG_VCNL4040_HPP
#define _PPG_VCNL4040_HPP

#include <cstdint>
#include <optional>

#include <zephyr/drivers/sensor.h>

#include "Device.hpp"

namespace Hardware
{
namespace Sensor
{
    class Vcnl4040
        : public Device
    {
        public:
            using ValueT = uint16_t;
        public:
            Vcnl4040(const device* const dev);
            std::optional<ValueT> getProximity();
    };
}
}

#endif //_PPG_VCNL4040_HPP