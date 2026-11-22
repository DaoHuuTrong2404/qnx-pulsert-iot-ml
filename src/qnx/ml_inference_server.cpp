#include "rta_auditor.hpp"
#include "telemetry_protocol.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#ifndef QNX_POSIX_HOST_SIM
#include <pthread.h>
#include <sched.h>
#include <sys/neutrino.h>
#else
#include <unistd.h>
#endif

namespace {

std::uint64_t monotonic_ns() {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

void print_rta_startup_audit() {
  std::vector<pulsert::RtTaskSpec> tasks = {
      {"sensor_ingest_fifo", 40, 220.0, 2000.0, 2000.0, 45.0},
      {"qnx_ml_infer_srv", 30, 380.0, 2000.0, 2000.0, 45.0},
      {"aps_telemetry_log", 12, 950.0, 10000.0, 10000.0, 0.0}};
  const auto rep = pulsert::analyze_schedulability(tasks);
  std::printf(
      "[QNX-PulseRT] RTA Audit: U=%.4f (LL_bound=%.4f) exact_schedulable=%s\n",
      rep.total_utilization, rep.liu_layland_bound,
      rep.exact_schedulable ? "TRUE" : "FALSE");
  for (const auto &t : rep.tasks) {
    std::printf("  - %-20s R_i=%.1fus <= D_i=%.1fus [%s]\n", t.name,
                t.response_us, t.deadline_us, t.schedulable ? "PASS" : "MISS");
  }
}

} // namespace

int main() {
  print_rta_startup_audit();

#ifndef QNX_POSIX_HOST_SIM
  // Lock real-time base priority; QNX MsgReceive inherits higher client
  // priority automatically.
  struct sched_param sp{};
  sp.sched_priority = 30;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
    std::fprintf(
        stderr,
        "[QNX-PulseRT] Warning: SCHED_FIFO priority setup failed (%s)\n",
        std::strerror(errno));
  }

  const int chid = ChannelCreate(0);
  if (chid == -1) {
    std::perror("[QNX-PulseRT] ChannelCreate failed");
    return 1;
  }
  std::printf("[QNX-PulseRT] Neutrino IPC server listening on chid=%d "
              "(zero-heap loop)\n",
              chid);

  pulsert::SensorFrameMsg req{};
  pulsert::InferenceReplyMsg rep{};
  struct _msg_info info{};

  while (true) {
    const int rcvid = MsgReceive(chid, &req, sizeof(req), &info);
    if (rcvd == -1) {
      if (errno == EINTR)
        continue;
      std::perror("[QNX-PulseRT] MsgReceive error");
      break;
    }
    if (rcvd == 0) {
      // Pulse received (ignored in deterministic frame handler)
      continue;
    }

    const std::uint64_t t_start = monotonic_ns();
    if (req.magic != pulsert::kProtocolMagic) {
      MsgError(rcvd, EINVAL);
      continue;
    }

    const std::int32_t score_q15 =
        pulsert::run_q15_mlp_inference(req.features_q15);
    const std::uint64_t t_end = monotonic_ns();
    const std::uint32_t exec_us =
        static_cast<std::uint32_t>((t_end - t_start) / 1000ull);

    std::memset(&rep, 0, sizeof(rep));
    rep.magic = pulsert::kProtocolMagic;
    rep.seq = req.seq;
    rep.t_reply_ns = t_end;
    rep.exec_us = exec_us;
    rep.anomaly_score_q15 = score_q15;
    rep.alert_tier =
        static_cast<std::uint8_t>(pulsert::classify_anomaly_q15(score_q15));
    rep.deadline_met = exec_us <= 2000u ? 1u : 0u;

    MsgReply(rcvd, EOK, &rep, sizeof(rep));
  }

  ChannelDestroy(chid);
  return 0;
#else
  // Portable POSIX self-test simulation (runs deterministic 10-frame
  // verification)
  std::printf("[QNX-PulseRT] POSIX Host Sim Mode: running 10 deterministic "
              "frames...\n");
  for (std::uint64_t seq = 1; seq <= 10; ++seq) {
    pulsert::SensorFrameMsg req{};
    req.magic = pulsert::kProtocolMagic;
    req.msg_type = pulsert::kMsgTypeSensorFrame;
    req.sensor_id = 1;
    req.seq = seq;
    req.t_capture_ns = monotonic_ns();
    const std::int16_t amp =
        static_cast<std::int16_t>(seq >= 8 ? 26000 : 4000 + seq * 1200);
    req.features_q15[0] = amp;
    req.features_q15[1] = static_cast<std::int16_t>(amp / 2);
    req.features_q15[2] = static_cast<std::int16_t>(amp / 3);
    req.features_q15[3] = amp;

    const std::uint64_t t0 = monotonic_ns();
    const std::int32_t score = pulsert::run_q15_mlp_inference(req.features_q15);
    const std::uint64_t t1 = monotonic_ns();
    const auto tier =
        static_cast<unsigned>(pulsert::classify_anomaly_q15(score));
    std::printf("  seq=%llu score_q15=%d tier=%u exec_ns=%llu\n",
                static_cast<unsigned long long>(seq), score, tier,
                static_cast<unsigned long long>(t1 - t0));
  }
  return 0;
#endif
}
