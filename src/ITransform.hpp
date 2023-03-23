#ifndef _PPG_ITRANSFORM_HPP
#define _PPG_ITRANSFORM_HPP

#include <array>
#include <cstddef>

namespace Dsp
{
    template <
        typename SampleT, typename TransformT
        , size_t InputLen, size_t OutputLen = InputLen
    >
    class ITransform
    {
        public:
            using InputT = std::array<SampleT, InputLen>;
            using OutputT = std::array<TransformT, OutputLen>;
            virtual OutputT transform(InputT& input) = 0;
            OutputT operator()(InputT& input)
            {
                return transform(input);
            }
    };
}

#endif //_PPG_ITRANSFORM_HPP