#include <cstdio>

#include <zephyr/device.h>
#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(ppg);

#include "Application.hpp"

Application::Application()
    : prox_{DEVICE_DT_GET_ONE(vishay_vcnl4040)}
    , neopixel_{DEVICE_DT_GET(DT_ALIAS(neopixel))}
    , serial_{DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart)}
    , ppg_{prox_}
    , hr_{sampleRate_}
    , dataCollector_{ppg_}
{ }

bool Application::run()
{
    // init
    LOG_INF("Initializing hardware...");
    auto status = init();
    if(!status) return false;
    LOG_INF("Hardware initialization successful");
    // main loop
    while(true)
    {
        using namespace std::chrono_literals;
        if(!serial_.isOpen())
        {
            dataCollector_.stop();
            neopixel_.setColor(Color::Color{10, 0, 0});
            // wait for DTR
            LOG_INF("Waiting for USB connection");
            while(!serial_.isOpen())
            {
                k_msleep(100);
            }
            neopixel_.setColor(Color::Color{0, 10, 0});
            LOG_INF("USB connected");
            dataCollector_.start(sampleTime_);
        }
        else
        {
            const auto ppgMeasurement = ppg_.getMeasurement(10ms);
            if(ppgMeasurement.has_value())
            {
                auto bpm = hr_.process(ppgMeasurement.value());
                auto len = snprintf(serialBuf_, sizeof(serialBuf_)
                                , "%" PRIu64 ",%d,%d,%d\r\n"
                                , ppgMeasurement.value().timestamp
                                , ppgMeasurement.value().raw
                                , ppgMeasurement.value().filtered
                                , bpm);
                serial_.write(reinterpret_cast<std::byte*>(serialBuf_), len);
            }
        }
    }
    return true; //< should never reach this point
}

bool Application::init()
{
    if (!prox_.isReady())
    {
        LOG_ERR("Proximity sensor not ready.");
        return false;
    }

    if (!neopixel_.isReady())
    {
        LOG_ERR("Neopixel not ready.");
        return false;
    }

    if(!serial_.enable())
    {
        LOG_ERR("Can't enable serial via USB CDC.");
        return false;
    }
    return true;
}