# QNX-PulseRT — Deterministic Neutrino IPC & Q15 TinyML Scheduler (`QNX Global Hackathon 2026`)

Zero-heap, fixed-point **Q15 TinyML anomaly detection** running across native **QNX SDP 8.0 Neutrino synchronous IPC** (`ChannelCreate` / `MsgSend` / `MsgReceive` / `MsgReply`) under preemptive `SCHED_FIFO` + Priority Inheritance, paired with a formal **Fixed-Point Response-Time Analysis (RTA)** auditor and an interactive browser workstation.

## Architecture Overview

1. **Zero-Allocation Telemetry Protocol (`src/qnx/telemetry_protocol.hpp`)**:
   - Fixed binary IPC packet (`SensorFrameMsg`, `AnomalyReplyMsg`) with CRC-16 CCITT integrity verification and exact Q1.15 integer MLP weights (`W1`, `B1`, `W2`, `B2` — zero runtime `malloc` or floating-point operations inside the hot inference loop).
2. **Exact Fixed-Point RTA Auditor (`src/qnx/rta_auditor.hpp`)**:
   - Evaluates CPU utilization `U`, Liu-Layland bound, and exact fixed-point worst-case response times `R_i` accounting for QNX Priority Inheritance blocking (`B_i = 45 us` across higher/intermediate tasks).
   - Nominal 3-task set (`sensor_ingest_fifo` Prio 40 `C=220us, T=2000us`; `qnx_ml_infer_srv` Prio 30 `C=380us, T=2000us`; `aps_telemetry_log` Prio 12 `C=950us, T=10000us`) yields:
     - `U = 0.3950 <= LL = 0.7798`
     - Exact RTA worst-case response times: `R = [265, 645, 1550] us` (`0` deadline misses).
3. **Neutrino Server & Ingest Client (`src/qnx/ml_inference_server.cpp`, `src/qnx/sensor_ingest.cpp`)**:
   - Conditional `#ifdef __QNXNTO__` paths invoke real QNX Neutrino `ChannelCreate`, `ConnectAttach`, `MsgSend`, `MsgReceive`, `MsgReply`, and `sched_setscheduler(..., SCHED_FIFO, ...)` calls, with a portable POSIX fallback stub for cross-host dry runs.
4. **Interactive QNX RTOS Simulator (`docs/index.html`)**:
   - Live 2.5 ms `SCHED_FIFO` + Priority Inheritance Gantt canvas, real-time Q15 TinyML telemetry waveform, interactive WCET/mutex blocking sliders, and exact 1-microsecond hyperperiod simulator.

## Verification & Build

- **Formal RTA + Q15 Error Proof (`verify_qnx_scheduler.py` via `lambai_overlord.py`)**:
  - Exact RTA `[265, 645, 1550]` matches 1-us discrete event simulation `[265, 645, 1550]` (`0` nominal misses).
  - Q15 fixed-point MLP vs. IEEE-754 float64 reference max absolute error: `0.000181`.
  - Priority Inversion Storm (`C0=820us, C1=1050us, B=240us`) triggers `1` simulated deadline miss and non-schedulable RTA verdict as expected.
- **Browser Simulator JS Parity (`tests/verify_qnx_js_parity.mjs`)**:

```bash
node --test tests/verify_qnx_js_parity.mjs
```

- **QNX Target Build (inside QNX SDP 8.0 / Docker container with `qcc`)**:
  *(Note: Not compiled locally on Windows host because QNX SDP / `g++` is not installed in the local Windows environment.)*

```bash
source ~/qnx800/qnxsdp-env.sh
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=qcc ..
cmake --build .
```
