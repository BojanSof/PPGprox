#ifndef _PPG_COLOR_HPP
#define _PPG_COLOR_HPP

#include <cstdint>
#include <string_view>

namespace Color
{
    class Color
    {
    public:
        constexpr Color(const uint8_t red = 0, const uint8_t green = 0, const uint8_t blue = 0)
            : r_{red}, g_{green}, b_{blue}
        {
        }

        void set(const uint8_t red = 0, const uint8_t green = 0, const uint8_t blue = 0);
        constexpr uint8_t getRed() const { return r_; }
        constexpr uint8_t getGreen() const { return g_; }
        constexpr uint8_t getBlue() const { return b_; }

    private:
        uint8_t r_, g_, b_;
    };
}

#endif //_PPG_COLOR_HPP