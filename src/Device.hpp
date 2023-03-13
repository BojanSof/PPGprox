#ifndef _PPG_DEVICE_HPP
#define _PPG_DEVICE_HPP

#include <zephyr/device.h>

namespace Hardware
{
    class Device
    {
        public:
            Device(const device* const dev);
            bool isReady() const;
            const device* getDevicePointer();
        protected:
            const device* const device_;
    };
}


#endif //_PPG_DEVICE_HPP