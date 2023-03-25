#ifndef _PPG_MOVING_AVERAGE_FILTER_HPP
#define _PPG_MOVING_AVERAGE_FILTER_HPP

#include <array>
#include <cstddef>

#include "IFilter.hpp"

namespace Dsp
{
    template <size_t NumSamples, typename SampleT, typename FilteredT = SampleT>
    class MovingAverageFilter
        : public IFilter<SampleT, FilteredT>
    {
        public:
            MovingAverageFilter()
                : samples_{}
                , iTail_{}
                , sum_{}
            { }

            FilteredT apply(const SampleT& sample)
            {
                auto oldestSample = samples_[iTail_];
                sum_ -= oldestSample;
                sum_ += sample;
                samples_[iTail_] = sample;
                iTail_ = (iTail_ == samples_.size()) ? 0 : iTail_ + 1;
                return sum_ / NumSamples;
            }            
        private:
            std::array<SampleT, NumSamples> samples_;
            size_t iTail_;
            SampleT sum_;
    };
}

#endif // _PPG_MOVING_AVERAGE_FILTER_HPP