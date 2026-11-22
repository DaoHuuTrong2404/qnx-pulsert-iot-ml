cmake_minimum_required(VERSION 3.16)
project(qnx_pulsert LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QNX_POSIX_HOST_SIM "Build portable POSIX simulation mode when not using QNX SDP 8.0" OFF)

add_executable(ml_inference_server src/qnx/ml_inference_server.cpp)
add_executable(sensor_ingest src/qnx/sensor_ingest.cpp)

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  target_compile_options(ml_inference_server PRIVATE -Wall -Wextra -O2)
  target_compile_options(sensor_ingest PRIVATE -Wall -Wextra -O2)
elseif(QNX_POSIX_HOST_SIM OR NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
  target_compile_definitions(ml_inference_server PRIVATE QNX_POSIX_HOST_SIM=1)
  target_compile_definitions(sensor_ingest PRIVATE QNX_POSIX_HOST_SIM=1)
  find_package(Threads REQUIRED)
  target_link_libraries(ml_inference_server PRIVATE Threads::Threads)
  target_link_libraries(sensor_ingest PRIVATE Threads::Threads)
endif()
