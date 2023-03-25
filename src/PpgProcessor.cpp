#include "PpgProcessor.hpp"

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(ppg);

namespace Processor
{
    Ppg::Ppg(Ppg::Proximity& sensor)
        : sensor_{sensor}
        , filter_{{ 0.13672873f, 0.0f, -0.13672873f, 1.705965f, -0.72654253f }}
    {
        k_msgq_init(&queue_, queueBuffer_, sizeof(Measurement), queueSize_);
    }

    bool Ppg::measure(const uint64_t& timestamp)
    {
        Measurement measurement{};
        // create new measurement
        auto proximity = sensor_.getProximity();
        if(proximity.has_value())
        {
            measurement.raw = proximity.value();
            measurement.filtered = filter_(measurement.raw);
        }
        else
        {
            return false;
        }
        measurement.timestamp = timestamp;
        // put measurement in queue
        if(k_msgq_put(&queue_, &measurement, K_NO_WAIT) != 0)
        {
            LOG_WRN("Queue is full. Sample dropped.");
            return false;
        }
        return true;
    }

    std::optional<Ppg::Measurement> Ppg::getMeasurement(const std::chrono::milliseconds& timeout)
    {
        Measurement measurement;
        if(k_msgq_get(&queue_, &measurement, K_MSEC(timeout.count())) == 0)
        {
            return measurement;
        }
        return {};
    }
}