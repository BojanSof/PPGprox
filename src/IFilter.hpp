#ifndef _PPG_IFILTER_HPP
#define _PPG_IFILTER_HPP

namespace Dsp
{
    template <typename SampleT, typename FilteredT = SampleT>
    class IFilter
    {
        public:
            virtual FilteredT apply(const SampleT& sample) = 0;
            FilteredT operator()(const SampleT& sample)
            {
                return apply(sample);
            }
    };
}

#endif //_PPG_IFILTER_HPP