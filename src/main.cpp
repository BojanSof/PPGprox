#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "Timer.hpp"
#include "Vcnl4040.hpp"
#include "Ws2812b.hpp"
#include "Serial.hpp"

LOG_MODULE_REGISTER(ppg_using_proximity);

int main()
{
    using Proximity = Hardware::Sensor::Vcnl4040;
    using Neopixel = Hardware::Ws2812b;
    using Timer = Hardware::Timer;
    using Serial = Hardware::Serial;

    Proximity prox{DEVICE_DT_GET_ONE(vishay_vcnl4040)};
    Neopixel neopix{DEVICE_DT_GET(DT_ALIAS(neopixel))};
    Serial serial{DEVICE_DT_GET(DT_CHOSEN(zephyr_console))};

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

    neopix.setColor(Color::Color{10, 0, 0});

    // wait for DTR
    while(!serial.isDTRset())
    {
        LOG_INF("Waiting for DTR");
        k_msleep(100);
    }

    neopix.setColor(Color::Color{0, 10, 10});

    Timer sampleTimer{};
    using namespace std::chrono_literals;
    uint16_t proxVal{};
    bool newVal{};

    sampleTimer.start(20ms, [&prox, &proxVal, &newVal]{
        // LOG_INF("TIMER EXPIRED!!!");
        proxVal = prox.getProximity().value_or(proxVal);
        newVal = true;
    });

    while (1)
    {
        if(newVal)
        {
            LOG_INF("Proximity value: %d\n", proxVal);
            printk("%d\r\n", proxVal);
            newVal = false;
        }
        k_msleep(5);
    }
}
