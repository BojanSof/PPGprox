#ifndef _PPG_HR_PROCESSOR_HPP
#define _PPG_HR_PROCESSOR_HPP

#include <zephyr/logging/log.h>

#include <array>
#include <cstdint>
#include "Fft.hpp"

namespace Processor
{
    template <size_t NumSamples, size_t FftLength = 256, typename SampleT = int16_t>
    class HeartRate
    {
        private:
            static constexpr size_t iSampleStart = (FftLength - NumSamples) / 2;
            static constexpr size_t iSampleEnd = FftLength - iSampleStart;
        public:
            HeartRate(const size_t fs)
                : samples_{}
                , iSample_{}
                , fs_{fs}
                , fft_{}
                , bpm_{}
            {
                static_assert(FftLength >= NumSamples, "FFT length must be >= than NumSamples");
            }

            uint8_t process(const SampleT& sample)
            {
                samples_[iSample_++] = sample;
                if(iSample_ == NumSamples)
                {
                    // calculate fft
                    auto fftMag = fft_.getMagnitudeSqr(samples_);
                    // LOG_HEXDUMP_WRN(fftMag.data(), fftMag.size() * sizeof(fftMag[0]), "FFT MAG");
                    // find frequency of max fft value
                    size_t iFftMax{0};
                    float32_t fftMax{fftMag[0]};
                    for(size_t iFft{0}; iFft < fftMag.size(); ++iFft)
                    {
                        if(fftMag[iFft] > fftMax)
                        {
                            fftMax = fftMag[iFft];
                            iFftMax = iFft;
                        }
                    }
                    // convert maxIndex to frequency and calculate bpm
                    // bpm_ = (60 * (fs_/2) * iFftMax) / 128;
                    // static constexpr uint8_t bpmMaxVal = 195;
                    // static constexpr uint8_t bpmMinVal = 35;
                    // bpm_ = std::min(bpm_, bpmMaxVal);
                    // bpm_ = std::max(bpm_, bpmMinVal);
                    bpm_ = iFftMax;
                    iSample_ = 0;
                }
                return bpm_;
            }
        private:
            std::array<float32_t, FftLength> samples_;
            size_t iSample_;
            const size_t fs_;
            Dsp::Fft<FftLength> fft_;
            uint8_t bpm_;
    };
}

#endif //_PPG_HR_PROCESSOR_HPP