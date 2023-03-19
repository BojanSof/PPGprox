#include "Vcnl4040.hpp"

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(ppg_using_proximity, LOG_LEVEL_DBG);

namespace Hardware
{
namespace Sensor
{
    Vcnl4040::Vcnl4040(const device* const dev)
        : Device{dev}
    { }

    std::optional<Vcnl4040::ValueT> Vcnl4040::getProximity()
    {
        const auto dev = getDevicePointer();
        if(sensor_sample_fetch(dev) < 0)
        {
            LOG_DBG("Can't fetch new proximity sample");
            return {};
        }
        sensor_value sensorValue{};
        if(sensor_channel_get(dev, SENSOR_CHAN_PROX, &sensorValue) < 0)
        {
            LOG_DBG("Can't get new proximity sample");
            return {};
        }
        else
        {
            return static_cast<ValueT>(sensorValue.val1);
        }
    }
}
}