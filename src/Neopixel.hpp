#ifndef _PPG_NEOPIXEL_HPP
#define _PPG_NEOPIXEL_HPP

#include "Device.hpp"
#include "Color.hpp"

namespace Hardware
{
    class Neopixel
        : public Device
    {
        public:
            Neopixel(const device* const dev);
            void setColor(const Color::Color& c);
    };
}

#endif //_PPG_NEOPIXEL_HPP