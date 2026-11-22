# QNX-PulseRT — Devpost Submission (`QNX Global Hackathon 2026 — IoT & Machine Learning/AI`)

## Inspiration
Industrial edge AI workloads on automotive and smart-factory controllers often fail real-time determinism when Python/dynamic memory runtimes cause unpredictable tail latencies or priority inversions. We set out to build an end-to-end **deterministic TinyML inference pipeline** rooted in QNX Neutrino RTOS semantics — proving both numerically and visually that AI inference can sit cleanly inside microsecond-level `SCHED_FIFO` budgets.

## What it does
**QNX-PulseRT** pairs a zero-heap Q15 Fixed-Point MLP anomaly classifier over native QNX Neutrino synchronous message passing (`ChannelCreate`/`MsgSend`/`MsgReceive`/`MsgReply`) with an embedded **Fixed-Point Response-Time Analysis (RTA)** auditor and interactive browser telemetry workstation:
- **Deterministic Edge ML Server (`qnx_ml_infer_srv`):** Classifies 4-channel edge telemetry (Vibration RMS, Bearing Temperature, Current Harmonics, Acoustic Kurtosis) using pure integer Q1.15 multiply-shifts and saturating ReLU inside a constant execution envelope (`WCET = 380 us`).
- **Preemptive RTOS Auditor:** Models QNX Priority Inheritance across a shared mutex (`B = 45 us`), verifying `U = 0.3950`, Liu-Layland bound `0.7798`, and exact worst-case response times `[265, 645, 1550] us` across a 10 ms hyperperiod with zero misses.
- **Live Fault & Storm Workstation (`docs/index.html`):** Lets operators inject thermal drift, bearing faults, or priority-inversion storms and immediately inspect the 1-us preemptive Gantt timeline, PI priority elevation bars, Q15 waveform, and RTA verdict.

## How we built it
- **C++17 RTOS Core (`src/qnx/`):** Authored `#ifdef __QNXNTO__` Neutrino IPC client/server code alongside deterministic CRC-16 telemetry framing and fixed-point RTA verification headers.
- **Dual Formal Verification:** Cross-checked the C++ constants against both a standalone 1-us Python hyperperiod simulator (`verify_qnx_scheduler.py`, `q15_max_abs_err = 0.000181`) and a Node parity suite (`tests/verify_qnx_js_parity.mjs`) directly testing `docs/index.html`'s JavaScript scheduler.

## Challenges we ran into
During cross-verification between the analytical RTA solver and our 1-microsecond discrete event simulator, an initial draft underestimated blocking for intermediate-priority `qnx_ml_infer_srv` (`20 us` instead of `45 us`), causing simulation `R_1 = 665 us > 620 us`. Accounting accurately for QNX Priority Inheritance — where elevating a low-priority mutex holder to Priority 40 delays every intermediate priority task by the full `45 us` critical section — brought analytical RTA and microsecond simulation into exact agreement (`[265, 645, 1550] == [265, 645, 1550]`).

## Accomplishments that we're proud of
- Microsecond-exact agreement between closed-form Fixed-Point RTA (`[265, 645, 1550]`) and 1-us event simulation (`[265, 645, 1550]`).
- Quantization fidelity within `0.000181` max absolute error using pure integer Q15 arithmetic with zero heap allocations.
- Real-time interactive Gantt & Q15 oscilloscope workstation running standalone in any browser.

## What we learned
Deterministic ML at the edge is as much a scheduling discipline as a quantization problem — Priority Inheritance blocking must be propagated rigorously through intermediate priority tiers in RTA proofs.

## What's next for QNX-PulseRT
Compiling and profiling directly inside QNX SDP 8.0 target hardware (Raspberry Pi 4 / QEMU x86_64) via `tracelogger` + QNX System Profiler event capture.
