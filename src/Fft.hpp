#ifndef _PPG_FFT_HPP
#define _PPG_FFT_HPP

#include <cstdint>
#include <cmath>
#include <complex>

#include <arm_math.h>

#include "ITransform.hpp"

namespace Dsp
{
    template <uint16_t Length>
    class Fft
        : public ITransform<float32_t, std::complex<float32_t>, Length, Length/2>
    {
        public:
            using InputT = ITransform<float32_t, std::complex<float32_t>, Length, Length/2>::InputT;
            using OutputT = ITransform<float32_t, std::complex<float32_t>, Length, Length/2>::OutputT;
            using MagnitudeT = std::array<float32_t, Length/2>;
            using AngleT = std::array<float32_t, Length/2>;

            Fft()
            {
                static_assert(Length && !(Length & (Length - 1)), "FFT Length must be power of 2");
                arm_rfft_fast_init_f32(&inst_, Length);
            }

            OutputT transform(const InputT& input) override
            {
                OutputT fft{};
                const auto fftInterleaved = transformInterleaved(input);
                using namespace std::complex_literals;
                for(size_t iFft = 0; iFft < fftInterleaved.size() /2 - 1; ++iFft)
                {
                    fft[iFft] = fftInterleaved[2 * iFft] + 1i * fftInterleaved[2 * iFft + 1];
                }
                return fft;
            }

            MagnitudeT getMagnitudeSqr(const InputT& input)
            {
                MagnitudeT mag{};
                const auto fftInterleaved = transformInterleaved(input);
                arm_cmplx_mag_squared_f32(fftInterleaved.data(), mag.data(), mag.size());
                return mag;
            }

            AngleT getAngleRad(const InputT& input)
            {
                AngleT angle{};
                const auto fftInterleaved = transformInterleaved(input);
                for(size_t iFft = 0; iFft < fftInterleaved.size() /2 - 1; ++iFft)
                {
                    angle[iFft] = std::atan2(fftInterleaved[2 * iFft], fftInterleaved[2 * iFft + 1]); 
                }
                return angle;
            }

        private:
            InputT transformInterleaved(const InputT& input)
            {
                InputT transformed{};
                arm_rfft_fast_f32(&inst_, input.data(), transformed.data(), 0);
                return transformed;
            }
        private:
            arm_rfft_fast_instance_f32 inst_;
    };
}

#endif //_PPG_FFT_HPP