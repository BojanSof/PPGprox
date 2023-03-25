#include <cstdio>

#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>

LOG_MODULE_REGISTER(ppg, LOG_LEVEL_DBG);

#include "Timer.hpp"
#include "Proximity.hpp"
#include "Neopixel.hpp"
#include "Serial.hpp"

#include "PpgProcessor.hpp"
#include "HrProcessor.hpp"


int main()
{
    using Proximity = Hardware::Proximity;
    using Neopixel = Hardware::Neopixel;
    using Timer = Hardware::Timer;
    using Serial = Hardware::Serial;
    using HeartRate = Processor::HeartRate<100>;
    using Ppg = Processor::Ppg;

    Proximity prox{DEVICE_DT_GET_ONE(vishay_vcnl4040)};
    Neopixel neopix{DEVICE_DT_GET(DT_ALIAS(neopixel))};
    Serial serial{DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart)};

    static constexpr std::size_t serialBufSize = 64;
    char serialBuf[serialBufSize]{};

    if (!prox.isReady())
    {
        LOG_ERR("Proximity sensor not ready.\n");
        return -1;
    }

    if (!neopix.isReady())
    {
        LOG_ERR("Neopixel is not ready.\n");
        return -1;
    }

    if(!serial.enable())
    {
        LOG_ERR("Can't enable serial via USB CDC.\n");
        return -1;
    }

    Ppg ppg{prox};
    HeartRate hr{50};



    Timer sampleTimer{};
    auto sampleTimerCallback = [&ppg]() mutable {
        auto timestamp = static_cast<uint64_t>(k_cycle_get_32()) * 1000000U
                        / sys_clock_hw_cycles_per_sec();
        if(!ppg.measure(timestamp))
        {
            LOG_WRN("Couldn't measure ppg");
        }
    };

    using namespace std::chrono_literals;
    while (1)
    {
        if(!serial.isOpen())
        {
            sampleTimer.stop();
            neopix.setColor(Color::Color{10, 0, 0});
            // wait for DTR
            LOG_INF("Waiting for DTR");
            while(!serial.isOpen())
            {
                k_msleep(100);
            }
            LOG_INF("DTR set");
            neopix.setColor(Color::Color{0, 10, 0});
            sampleTimer.start(20ms, sampleTimerCallback);
        }
        else
        {
            const auto ppgMeasurement = ppg.getMeasurement(10ms);
            if(ppgMeasurement.has_value())
            {
                auto bpm = hr.process(ppgMeasurement.value().filtered);
                auto len = snprintf(serialBuf, serialBufSize
                                , "%" PRIu64 ",%d,%d,%d\r\n"
                                , ppgMeasurement.value().timestamp
                                , ppgMeasurement.value().raw
                                , ppgMeasurement.value().filtered
                                , bpm);
                serial.write(reinterpret_cast<std::byte*>(serialBuf), len);
            }
        }
    }
}
