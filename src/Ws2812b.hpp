#ifndef _PPG_WS2812B_HPP
#define _PPG_WS2812B_HPP

#include "Device.hpp"
#include "Color.hpp"

namespace Hardware
{
    class Ws2812b
        : public Device
    {
        public:
            Ws2812b(const device* const dev);
            void setColor(const Color::Color& c);
    };
}

#endif //_PPG_WS2812B_HPP