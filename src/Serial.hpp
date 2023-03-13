#ifndef _PPG_SERIAL_HPP
#define _PPG_SERIAL_HPP

#include "Device.hpp"

namespace Hardware
{
    class Serial
        : public Device
    {
        public:
            Serial(const device* const dev);
            bool enable();
            bool isDTRset();
    };
}

#endif //_PPG_SERIAL_HPP