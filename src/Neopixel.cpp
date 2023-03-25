#include "Neopixel.hpp"

#include <zephyr/drivers/led_strip.h>

namespace Hardware
{
    Neopixel::Neopixel(const device* const dev)
        : Device{dev}
    { }

    void Neopixel::setColor(const Color::Color& c)
    {
        led_rgb code = {
            .r = c.getRed(),
            .g = c.getGreen(),
            .b = c.getBlue()
        };
        led_strip_update_rgb(getDevicePointer(), &code, 1);
    }
}