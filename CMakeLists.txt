# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

list(APPEND mcuboot_OVERLAY_CONFIG
  "${CMAKE_CURRENT_LIST_DIR}/mcuboot.conf"
)

list(APPEND DTC_OVERLAY_FILE
  "${CMAKE_CURRENT_LIST_DIR}/proximity.overlay"
  "${CMAKE_CURRENT_LIST_DIR}/usb_cdc.overlay"
)

set(INTERPROCEDURAL_OPTIMIZATION true)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_world)

target_include_directories(app PRIVATE
  "src"
)

target_sources(app PRIVATE
  "src/Color.hpp"
  "src/Color.cpp"
  "src/Timer.hpp"
  "src/Timer.cpp"
  "src/Device.hpp"
  "src/Device.cpp"
  "src/Serial.hpp"
  "src/Serial.cpp"
  "src/Vcnl4040.hpp"
  "src/Vcnl4040.cpp"
  "src/Ws2812b.hpp"
  "src/Ws2812b.cpp"
  "src/IFilter.hpp"
  "src/IIRFilter.hpp"
  "src/ITransform.hpp"
  "src/Fft.hpp"
  "src/HrProcessor.hpp"
  "src/main.cpp"
)