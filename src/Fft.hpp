#ifndef _PPG_FFT_HPP
#define _PPG_FFT_HPP

#include <cstdint>
#include <cmath>

#include <arm_math.h>

#include "ITransform.hpp"

namespace Dsp
{
    template <uint16_t Length>
    class Fft
        : public ITransform<float32_t, float32_t, Length, Length/2>
    {
        public:
            using InputT = ITransform<float32_t, float32_t, Length, Length/2>::InputT;
            using OutputT = ITransform<float32_t, float32_t, Length, Length/2>::OutputT;
            using MagnitudeT = std::array<float32_t, Length/2>;
            using AngleT = std::array<float32_t, Length/2>;

            Fft()
            {
                static_assert(Length && !(Length & (Length - 1)), "FFT Length must be power of 2");
                arm_rfft_fast_init_f32(&inst_, Length);
            }

            OutputT transform(InputT& input) override
            {
                OutputT transformed{};
                arm_rfft_fast_f32(&inst_, input.data(), transformed.data(), 0);
                return transformed;
            }

            MagnitudeT getMagnitudeSqr(InputT& input)
            {
                MagnitudeT mag{};
                auto fftInterleaved = transform(input);
                arm_cmplx_mag_squared_f32(fftInterleaved.data(), mag.data(), mag.size());
                return mag;
            }

            AngleT getAngleRad(InputT& input)
            {
                AngleT angle{};
                auto fftInterleaved = transform(input);
                for(size_t iFft = 0; iFft < fftInterleaved.size() /2 - 1; ++iFft)
                {
                    angle[iFft] = std::atan2(fftInterleaved[2 * iFft], fftInterleaved[2 * iFft + 1]); 
                }
                return angle;
            }

        private:
        private:
            arm_rfft_fast_instance_f32 inst_;
    };
}

#endif //_PPG_FFT_HPP