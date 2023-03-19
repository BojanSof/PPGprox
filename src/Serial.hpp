#ifndef _PPG_SERIAL_HPP
#define _PPG_SERIAL_HPP

#include <cstddef>

#include "Device.hpp"

namespace Hardware
{
    class Serial
        : public Device
    {
        public:
            Serial(const device* const dev);
            bool enable();
            bool isOpen();
            void write(const std::byte* const data, const std::size_t numBytes);
            void read(std::byte* data, const std::size_t numBytes);
    };
}

#endif //_PPG_SERIAL_HPP