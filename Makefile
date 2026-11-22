# QNX SDP 8.0 & Portable POSIX Makefile for QNX-PulseRT
# Usage on QNX SDP 8.0 host (aarch64le Raspberry Pi 5 or x86_64 QEMU):
#   source ~/qnx800/qnxsdp-env.sh
#   make TARGET=aarch64le   (or TARGET=x86_64)
# Usage on POSIX host simulator:
#   make host

TARGET ?= x86_64
QCC ?= qcc -Vgcc_nto$(TARGET)
HOST_CXX ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

.PHONY: all host clean

all: ml_inference_server sensor_ingest

ml_inference_server: src/qnx/ml_inference_server.cpp src/qnx/telemetry_protocol.hpp src/qnx/rta_auditor.hpp
	$(QCC) $(CXXFLAGS) -o $@ $<

sensor_ingest: src/qnx/sensor_ingest.cpp src/qnx/telemetry_protocol.hpp src/qnx/rta_auditor.hpp
	$(QCC) $(CXXFLAGS) -o $@ $<

host:
	$(HOST_CXX) $(CXXFLAGS) -DQNX_POSIX_HOST_SIM=1 -pthread -o ml_inference_server_sim src/qnx/ml_inference_server.cpp
	$(HOST_CXX) $(CXXFLAGS) -DQNX_POSIX_HOST_SIM=1 -pthread -o sensor_ingest_sim src/qnx/sensor_ingest.cpp

clean:
	rm -f ml_inference_server sensor_ingest ml_inference_server_sim sensor_ingest_sim
