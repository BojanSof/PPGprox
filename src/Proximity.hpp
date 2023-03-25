#ifndef _PPG_PROXIMITY_HPP
#define _PPG_PROXIMITY_HPP

#include <cstdint>
#include <optional>

#include <zephyr/drivers/sensor.h>

#include "Device.hpp"

namespace Hardware
{
    class Proximity
        : public Device
    {
        public:
            using ValueT = uint16_t;
        public:
            Proximity(const device* const dev);
            std::optional<ValueT> getProximity();
    };
}

#endif //_PPG_PROXIMITY_HPP