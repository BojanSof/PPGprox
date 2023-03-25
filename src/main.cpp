#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>

LOG_MODULE_REGISTER(ppg, LOG_LEVEL_DBG);

#include "Application.hpp"

int main()
{
    Application app;

    if(!app.run())
    {
        LOG_ERR("Can't run app.");
        return -1;
    }
}
