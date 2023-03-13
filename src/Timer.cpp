#include "Timer.hpp"

namespace Hardware
{
    std::array<Timer*, Timer::maxNumTimers> Timer::timers{};
    std::size_t Timer::nTimers{};
    std::size_t Timer::iTimerWorkqueue{};
    k_work Timer::work_ = Z_WORK_INITIALIZER(Timer::workqueueFunction);

    Timer::Timer()
        : iTimer{nTimers++}
    {
        assert(nTimers <= maxNumTimers);

        timers[iTimer] = this;
        k_timer_init(&timer_, &Timer::expiryFunction, NULL);
    }

    Timer::~Timer()
    {
        timers[iTimer] = nullptr;
        for (std::size_t iTim = iTimer + 1; iTim < nTimers; ++iTim)
        {
            timers[iTim - 1] = timers[iTim];
        }
        timers[nTimers - 1] = nullptr;
        nTimers--;
    }

    void Timer::expiryFunction(k_timer* timer)
    {
        for(std::size_t iTimer = 0; iTimer < nTimers; ++iTimer)
        {
            if(&(timers[iTimer]->timer_) == timer)
            {
                if(timers[iTimer]->callback_)
                {
                    if(timers[iTimer]->workqueue_)
                    {
                        iTimerWorkqueue = iTimer;
                        k_work_submit(&Timer::work_);
                    }
                    else
                    {
                        timers[iTimer]->callback_();
                    }
                }
            }
        }
    }

    void Timer::workqueueFunction(k_work* work)
    {
        if(timers[iTimerWorkqueue]->callback_)
        {
            timers[iTimerWorkqueue]->callback_();
        }
    }

}