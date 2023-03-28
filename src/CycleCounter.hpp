#ifndef _PPG_CYCLE_COUNTER_HPP
#define _PPG_CYCLE_COUNTER_HPP

#include <zephyr/kernel.h>

#include "Benchmark.hpp"

static constexpr uint32_t CpuFrequency = 64000000; //< 64 MHz

class CycleCounter
    : public Benchmark::ICycleCounter<
        uint32_t
        , CpuFrequency
    >
{
    public:
        CycleCounter()
        {
            // enable CYCCNT
            CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
            DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
        }

        time_point now() noexcept override
        {
            return time_point{ duration{ DWT->CYCCNT } };
        }
};

#endif //_PPG_CYCLE_COUNTER_HPP