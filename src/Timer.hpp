#ifndef _PPG_TIMER_HPP
#define _PPG_TIMER_HPP

#include <cassert>
#include <chrono>
#include <functional>

#include <zephyr/kernel.h>

namespace Hardware
{
    class Timer
    {
        private:
            using CallbackT = std::function<void()>;
        public:
            Timer();
            ~Timer();

            template< class Rep, class Period >
            void start(const std::chrono::duration<Rep, Period>& period, const CallbackT& callback, bool inIsr = false)
            {
                callback_ = callback;
                workqueue_ = !inIsr;
                const auto periodUs = std::chrono::microseconds(period).count();
                k_timer_start(&timer_, K_USEC(periodUs), K_USEC(periodUs));
            }

            void stop()
            {
                k_timer_stop(&timer_);
            }
        private:
            // for callbacks
            static constexpr std::size_t maxNumTimers = 20;
            static std::array<Timer*, maxNumTimers> timers;
            static std::size_t nTimers;
            static std::size_t iTimerWorkqueue;
            static k_work work_;
            static void expiryFunction(k_timer* timer);
            static void workqueueFunction(k_work* work);
        private:
            k_timer timer_;
            CallbackT callback_;
            std::size_t iTimer;
            bool workqueue_;
    };
}

#endif //_PPG_TIMER_HPP