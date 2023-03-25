#include "DataCollector.hpp"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(ppg);

DataCollector::DataCollector(Processor::Ppg& ppg)
    : ppg_{ppg}
{ }

void DataCollector::start(const std::chrono::milliseconds& samplingTime)
{
    sampleTimer_.start(samplingTime, [this] { collectData(); });
}

void DataCollector::stop()
{
    sampleTimer_.stop();
}

void DataCollector::collectData()
{
    auto timestamp = static_cast<uint64_t>(k_cycle_get_32()) * 1000000U
                    / sys_clock_hw_cycles_per_sec();
    if(!ppg_.measure(timestamp))
    {
        LOG_WRN("Couldn't measure ppg");
    }
}