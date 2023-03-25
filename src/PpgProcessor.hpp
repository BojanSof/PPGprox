#ifndef _PPG_PPG_PROCESSOR_HPP
#define _PPG_PPG_PROCESSOR_HPP

#include <chrono>
#include <cstdint>
#include <optional>

#include <zephyr/kernel.h>

#include "Proximity.hpp"
#include "IIRFilter.hpp"

namespace Processor
{
    class Ppg
    {
        using Proximity = Hardware::Proximity;
        public:
            struct Measurement
            {
                uint64_t timestamp;
                uint16_t raw;
                int16_t filtered;
            };
        public:
            Ppg(Proximity& sensor);
            bool measure(const uint64_t& timestamp);
            std::optional<Measurement> getMeasurement(const std::chrono::milliseconds& timeout);
        private:
            Proximity& sensor_;
            Dsp::IIRFilter<2> filter_;
            // message queue related
            static constexpr std::size_t queueSize_ = 10;
            char __aligned(4) queueBuffer_[queueSize_ * sizeof(Measurement)]{};
            k_msgq queue_;
    };
}

#endif //_PPG_PPG_PROCESSOR_HPP