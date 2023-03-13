#include "Color.hpp"

namespace Color
{
    void Color::set(const uint8_t red, const uint8_t green, const uint8_t blue)
    {
        r_ = red;
        g_ = green;
        b_ = blue;
    }
}