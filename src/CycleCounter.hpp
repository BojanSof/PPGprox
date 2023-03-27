#ifndef _PPG_CYCLE_COUNTER_HPP
#define _PPG_CYCLE_COUNTER_HPP

#include <zephyr/kernel.h>

#include "Benchmark.hpp"

class CycleCounter
    : public Benchmark::ICycleCounter<
        uint32_t
        , CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC
    >
{
    public:
        time_point now() noexcept override
        {
            return time_point{ duration{ k_cycle_get_32() } };
        }
};

#endif //_PPG_CYCLE_COUNTER_HPP