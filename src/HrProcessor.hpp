#ifndef _PPG_HR_PROCESSOR_HPP
#define _PPG_HR_PROCESSOR_HPP

#include <algorithm>
#include <array>
#include <cstdint>

#include "PpgProcessor.hpp"
#include "Fft.hpp"

namespace Processor
{
    template <
        size_t NumSamples
        , size_t NumSamplesHistory = NumSamples
        , size_t FftLength = 1024
    >
    class HeartRate
    {
        public:
            HeartRate(const uint16_t fs)
                : samples_{}
                , iSample_{}
                , fft_{}
                , fs_{fs}
                , bpm_{}
            {
                static_assert(FftLength >= NumSamples, "FFT length must be >= than NumSamples");
                static_assert(FftLength >= NumSamplesHistory, "FFT length must be >= than NumSamplesHistory");
                static_assert(NumSamplesHistory >= NumSamples, "FFT length must be >= than NumSamples");
            }

            uint8_t process(const Ppg::Measurement& measurement)
            {
                samples_[NumSamplesHistory - NumSamples + iSample_] = measurement.filtered;
                iSample_++;
                if(iSample_ == NumSamples)
                {
                    // calculate fft
                    auto fftMag = fft_.getMagnitudeSqr(samples_);
                    // find frequency of max fft value
                    const auto itrFftMax = std::max_element(fftMag.cbegin(), fftMag.cend());
                    const auto iFftMax = std::distance(fftMag.cbegin(), itrFftMax);
                    // convert maxIndex to frequency and calculate bpm
                    bpm_ = (60 * (fs_/2) * iFftMax) / (FftLength / 2);
                    // shift samples
                    for(size_t i = NumSamples; i < NumSamplesHistory; ++i)
                    {
                        samples_[i - NumSamples] = samples_[i];
                    }
                    iSample_ = 0;
                }
                return bpm_;
            }
        private:
            std::array<float32_t, FftLength> samples_;
            size_t iSample_;
            Dsp::Fft<FftLength> fft_;
            uint32_t fs_;
            uint8_t bpm_;
    };
}

#endif //_PPG_HR_PROCESSOR_HPP