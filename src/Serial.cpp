#include "Serial.hpp"

#include <zephyr/usb/usb_device.h>
#include <zephyr/drivers/uart.h>

namespace Hardware
{
    Serial::Serial(const device* const dev)
        : Device{dev}
    { }

    bool Serial::enable()
    {
        return !usb_enable(NULL);
    }

    bool Serial::isDTRset()
    {
        uint32_t dtr{};
        uart_line_ctrl_get(getDevicePointer(), UART_LINE_CTRL_DTR, &dtr);
        return dtr;
    }
}