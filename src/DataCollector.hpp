#ifndef _PPG_DATA_COLLECTOR_HPP
#define _PPG_DATA_COLLECTOR_HPP

#include <chrono>

#include "Timer.hpp"
#include "PpgProcessor.hpp"

class DataCollector
{
    public:
        DataCollector(Processor::Ppg& ppg);
        void start(const std::chrono::milliseconds& samplingTime);
        void stop();
    private:
        void collectData();
    private:
        Hardware::Timer sampleTimer_;
        Processor::Ppg& ppg_;
};

#endif //_PPG_DATA_COLLECTOR_HPP