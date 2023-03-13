#include "Vcnl4040.hpp"

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
            return {};
        }
        sensor_value sensorValue{};
        if(sensor_channel_get(dev, SENSOR_CHAN_PROX, &sensorValue) < 0)
        {
            return {};
        }
        else
        {
            return static_cast<ValueT>(sensorValue.val1);
        }
    }
}
}