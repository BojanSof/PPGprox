#ifndef _PPG_BENCHMARK_HPP
#define _PPG_BENCHMARK_HPP

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <utility> //< std::forward

namespace Benchmark
{
    template <typename Rep, size_t CoreFrequency>
    class ICycleCounter
    {
        public:
            using period = std::ratio<1, CoreFrequency>;
            using rep = Rep;
            using duration = std::chrono::duration<rep, period>;
            using time_point = std::chrono::time_point<ICycleCounter>;
            static constexpr bool is_steady = true;
        public:
            virtual time_point now() noexcept = 0;
    };

    template <typename CounterT, typename Func, typename... Args>
    decltype(auto) benchmark(CounterT& counter, Func&& f, Args&&... args)
    {
        auto startTime = counter.now();
        f(std::forward<Args>(args)...);
        auto endTime = counter.now();
        return endTime - startTime;
    }
}

#endif //_PPG_BENCHMARK_HPP