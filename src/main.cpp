#include <cstdio>

#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>

LOG_MODULE_REGISTER(ppg_using_proximity, LOG_LEVEL_DBG);

#include "Timer.hpp"
#include "Vcnl4040.hpp"
#include "Ws2812b.hpp"
#include "Serial.hpp"

#include "IIRFilter.hpp"
#include "HrProcessor.hpp"


int main()
{
    using Proximity = Hardware::Sensor::Vcnl4040;
    using Neopixel = Hardware::Ws2812b;
    using Timer = Hardware::Timer;
    using Serial = Hardware::Serial;
    using HeartRate = Processor::HeartRate<100>;

    Proximity prox{DEVICE_DT_GET_ONE(vishay_vcnl4040)};
    Neopixel neopix{DEVICE_DT_GET(DT_ALIAS(neopixel))};
    Serial serial{DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart)};
    HeartRate hr{50};

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

    struct Sample
    {
        uint64_t timestamp;
        uint16_t proximity;
    };

    static constexpr std::size_t sampleBufferSize = 10;
    char __aligned(4) sampleBuffer[sampleBufferSize * sizeof(Sample)]{};
    k_msgq sampleQueue{};
    k_msgq_init(&sampleQueue, sampleBuffer, sizeof(Sample), sampleBufferSize);

    Dsp::IIRFilter<2> proxFilter{
        { 0.13672873f, 0.0f, -0.13672873f, 1.705965f, -0.72654253f }
    };

    Timer sampleTimer{};
    auto sampleTimerCallback = [&prox, &sampleQueue, proxValue = uint16_t(0)]() mutable {
        proxValue = prox.getProximity().value_or(proxValue);
        auto timestamp = static_cast<uint64_t>(k_cycle_get_32()) * 1000000U / sys_clock_hw_cycles_per_sec();
        auto sample = Sample{timestamp, proxValue};
        if(k_msgq_put(&sampleQueue, &sample, K_NO_WAIT) != 0)
        {
            LOG_WRN("Queue is full. Sample dropped.");
        }
    };

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
            using namespace std::chrono_literals;
            sampleTimer.start(20ms, sampleTimerCallback);
        }
        else
        {
            Sample sample{};
            while(k_msgq_get(&sampleQueue, &sample, K_MSEC(5)) == 0)
            {
                int16_t proxFilteredVal = proxFilter(sample.proximity);
                auto bpm = hr.process(sample.proximity);
                auto len = snprintf(serialBuf, serialBufSize, "%" PRIu64 ",%d,%d,%d\r\n", sample.timestamp, sample.proximity, proxFilteredVal, bpm);
                serial.write(reinterpret_cast<std::byte*>(serialBuf), len);
            }
        }
    }
}
