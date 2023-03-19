#include "Serial.hpp"

#include <zephyr/kernel.h>
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

    void Serial::write(const std::byte* const data, const std::size_t numBytes)
    {
        for(std::size_t iByte = 0; iByte < numBytes; ++iByte)
        {
            uart_poll_out(getDevicePointer(), static_cast<uint8_t>(data[iByte]));
        }
    }
    
    void Serial::read(std::byte* data, const std::size_t numBytes)
    {
        for(std::size_t iByte = 0; iByte < numBytes; ++iByte)
        {
            while(uart_poll_in(getDevicePointer(), reinterpret_cast<uint8_t*>(data + iByte)))
            {
                k_yield();
            }
        }
    }
}