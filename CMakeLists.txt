cmake_minimum_required(VERSION 3.20.0)

list(APPEND mcuboot_OVERLAY_CONFIG
  "${CMAKE_CURRENT_LIST_DIR}/mcuboot.conf"
)

list(APPEND DTC_OVERLAY_FILE
  "${CMAKE_CURRENT_LIST_DIR}/proximity.overlay"
  "${CMAKE_CURRENT_LIST_DIR}/usb_cdc.overlay"
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_world)

target_include_directories(app PRIVATE
  "src"
)

if(CONFIG_BENCHMARK_CMSIS_DSP_CODE)
  target_sources(app PRIVATE
    "src/Device.hpp"
    "src/Device.cpp"
    "src/Serial.hpp"
    "src/Serial.cpp"
    "src/IFilter.hpp"
    "src/IIRFilter.hpp"
    "src/MovingAverageFilter.hpp"
    "src/ITransform.hpp"
    "src/Fft.hpp"
    "src/Benchmark.hpp"
    "src/CycleCounter.hpp"
    "src/BenchmarkDSP.cpp"
  )
else()
  target_sources(app PRIVATE
    "src/Color.hpp"
    "src/Color.cpp"
    "src/Timer.hpp"
    "src/Timer.cpp"
    "src/Device.hpp"
    "src/Device.cpp"
    "src/Serial.hpp"
    "src/Serial.cpp"
    "src/Proximity.hpp"
    "src/Proximity.cpp"
    "src/Neopixel.hpp"
    "src/Neopixel.cpp"
    "src/IFilter.hpp"
    "src/IIRFilter.hpp"
    "src/MovingAverageFilter.hpp"
    "src/ITransform.hpp"
    "src/Fft.hpp"
    "src/PpgProcessor.hpp"
    "src/PpgProcessor.cpp"
    "src/HrProcessor.hpp"
    "src/DataCollector.hpp"
    "src/DataCollector.cpp"
    "src/Application.hpp"
    "src/Application.cpp"
    "src/Benchmark.hpp"
    "src/CycleCounter.hpp"
    "src/main.cpp"
  )
endif(CONFIG_BENCHMARK_CMSIS_DSP_CODE)

set_property(TARGET app PROPERTY INTERPROCEDURAL_OPTIMIZATION True)