#include <cstdio>

#include <zephyr/logging/log.h>

#include "Timer.hpp"
#include "Vcnl4040.hpp"
#include "Ws2812b.hpp"
#include "Serial.hpp"

#include "IIRFilter.hpp"

LOG_MODULE_REGISTER(ppg_using_proximity, LOG_LEVEL_INF);

int main()
{
    using Proximity = Hardware::Sensor::Vcnl4040;
    using Neopixel = Hardware::Ws2812b;
    using Timer = Hardware::Timer;
    using Serial = Hardware::Serial;

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

    Timer sampleTimer{};
    using namespace std::chrono_literals;
    uint16_t proxVal{};
    int16_t proxFilteredVal{};
    uint64_t timestampUs{};
    bool newVal{};

    Dsp::IIRFilter<2> proxFilter{
        { 0.13672873f, 0.0f, -0.13672873f, 1.705965f, -0.72654253f }
    };

    auto sampleTimerCallback = [&prox, &proxVal, &timestampUs, &newVal]{
        proxVal = prox.getProximity().value_or(proxVal);
        timestampUs = static_cast<uint64_t>(k_cycle_get_32()) * 1000000U / sys_clock_hw_cycles_per_sec();
        newVal = true;
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
            sampleTimer.start(20ms, sampleTimerCallback);
        }
        else
        {
            if(newVal)
            {
                proxFilteredVal = proxFilter(proxVal);
                auto len = sprintf(serialBuf, "%" PRIu64 ",%d,%d\r\n", timestampUs, proxVal, proxFilteredVal);
                serial.write(reinterpret_cast<std::byte*>(serialBuf), len);
                newVal = false;
            }
            k_msleep(5);
        }
    }
}
