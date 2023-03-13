#include "Ws2812b.hpp"

#include <zephyr/drivers/led_strip.h>

namespace Hardware
{
    Ws2812b::Ws2812b(const device* const dev)
        : Device{dev}
    { }

    void Ws2812b::setColor(const Color::Color& c)
    {
        led_rgb code = {
            .r = c.getRed(),
            .g = c.getGreen(),
            .b = c.getBlue()
        };
        led_strip_update_rgb(getDevicePointer(), &code, 1);
    }
}