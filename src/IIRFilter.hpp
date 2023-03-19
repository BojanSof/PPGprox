#ifndef _PPG_IIRFILTER_HPP
#define _PPG_IIRFILTER_HPP

#include <algorithm>
#include <array>

#include <arm_math.h>

#include "IFilter.hpp"

namespace Dsp
{
    // IIR filter implementation using cascade of second-order Biquad sections
    template <size_t Order>
    class IIRFilter
        : public IFilter<float32_t, float32_t>
    {
        private:
            static constexpr size_t NumSections = Order / 2 + (Order % 2);
            static constexpr size_t NumCoeffsPerSection = 5;
            static constexpr size_t NumCoeffs = NumCoeffsPerSection * NumSections;
            using CoeffsT = std::array<float32_t, NumCoeffs>;

        public:
            IIRFilter(const CoeffsT& coeffs)
                : coeffs_{coeffs}, states_{}, inst_{NumSections, states_.data(), coeffs_.data()}
            {
                static_assert(Order > 0, "Filter order must be > 0");
            }

            float32_t apply(const float32_t& sample)
            {
                float32_t filteredSample{};
                apply(&sample, &filteredSample, 1);
                return filteredSample;
            }

            template<typename BlockT>
            void apply(const BlockT& in, BlockT& out)
            {
                apply(in.data(), out.data(), in.size());
            }

            template<typename BlockT>
            void operator()(const BlockT& in, BlockT& out)
            {
                apply(in, out);
            }
        private:
            void apply(const float32_t* in, const float32_t* out, const uint32_t blockSize)
            {
                arm_biquad_cascade_df2T_f32(&inst_, in, out, blockSize);
            }
        private:
            const CoeffsT coeffs_;
            static constexpr size_t NumStateVarsPerSection = 2;
            static constexpr size_t NumStateVars = NumStateVarsPerSection * NumSections;
            using StatesT std::array<float32_t, NumStateVarsPerSection * NumSections>;
            StatesT states_;
            arm_biquad_cascade_df2T_instance_f32 inst_; //< CMSIS-DSP filter instance
    };
}

#endif //_PPG_IIRFILTER_HPP