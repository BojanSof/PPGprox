#ifndef _PPG_APPLICATION_HPP
#define _PPG_APPLICATION_HPP

#include "Proximity.hpp"
#include "Neopixel.hpp"
#include "Serial.hpp"

#include "PpgProcessor.hpp"
#include "DataCollector.hpp"
#include "HrProcessor.hpp"

class Application
{
    public:
        Application();
        bool run();
    private:
        bool init();
    private:
        // hardware
        Hardware::Proximity prox_;
        Hardware::Neopixel neopixel_;
        Hardware::Serial serial_;
        // processors
        Processor::Ppg ppg_;
        static constexpr size_t hrSamples_ = 100;
        Processor::HeartRate<hrSamples_> hr_;
        DataCollector dataCollector_;
        // buffers, state vars, etc.
        static constexpr std::size_t serialBufSize_ = 64;
        char serialBuf_[serialBufSize_]{};
};

#endif //_PPG_APPLICATION_HPP