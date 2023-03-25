#include "Proximity.hpp"

namespace Hardware
{
    Proximity::Proximity(const device* const dev)
        : Device{dev}
    { }

    std::optional<Proximity::ValueT> Proximity::getProximity()
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