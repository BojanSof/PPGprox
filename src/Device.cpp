#include "Device.hpp"

namespace Hardware
{
    Device::Device(const device* const dev)
        : device_{dev}
    { }

    bool Device::isReady() const
    {
        return device_is_ready(device_);
    }

    const device* Device::getDevicePointer()
    {
        return device_;
    }
}