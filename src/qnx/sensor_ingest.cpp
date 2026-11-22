#include "telemetry_protocol.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#ifndef QNX_POSIX_HOST_SIM
#include <pthread.h>
#include <sched.h>
#include <sys/neutrino.h>
#endif

namespace {

std::uint64_t monotonic_ns() {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

void sleep_until_ns(std::uint64_t target_ns) {
  struct timespec ts{};
  ts.tv_sec = static_cast<time_t>(target_ns / 1000000000ull);
  ts.tv_nsec = static_cast<long>(target_ns % 1000000000ull);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) ==
         EINTR) {
  }
}

} // namespace

int main(int argc, char **argv) {
  const std::uint64_t period_ns = 2000000ull; // 2.0 ms = 500 Hz
  const std::uint64_t frames =
      (argc >= 2) ? std::strtoull(argv[1], nullptr, 10) : 250ull;

#ifndef QNX_POSIX_HOST_SIM
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s <frames> <server_pid> <server_chid>\n",
                 argv[0]);
    return 1;
  }
  const pid_t srv_pid = static_cast<pid_t>(std::atoi(argv[2]));
  const int srv_chid = std::atoi(argv[3]);

  struct sched_param sp{};
  sp.sched_priority = 40;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);

  const int coid = ConnectAttach(0, srv_pid, srv_chid, _NTO_SIDE_CHANNEL, 0);
  if (coid == -1) {
    std::perror("[sensor_ingest] ConnectAttach failed");
    return 1;
  }

  std::uint64_t next_tick = monotonic_ns() + period_ns;
  std::uint64_t misses = 0;
  pulsert::SensorFrameMsg msg{};
  pulsert::InferenceReplyMsg rep{};

  for (std::uint64_t seq = 1; seq <= frames; ++seq) {
    sleep_until_ns(next_tick);
    const std::uint64_t t_now = monotonic_ns();

    msg.magic = pulsert::kProtocolMagic;
    msg.msg_type = pulsert::kMsgTypeSensorFrame;
    msg.sensor_id = 1;
    msg.seq = seq;
    msg.t_capture_ns = t_now;
    const std::int16_t base = static_cast<std::int16_t>(
        (seq % 60 >= 50) ? 25000 : 5000 + (seq % 10) * 800);
    msg.features_q15[0] = base;
    msg.features_q15[1] = static_cast<std::int16_t>(base / 2);
    msg.features_q15[2] = static_cast<std::int16_t>(base / 3);
    msg.features_q15[3] = base;

    if (MsgSend(coid, &msg, sizeof(msg), &rep, sizeof(rep)) == -1) {
      std::perror("[sensor_ingest] MsgSend failed");
      break;
    }
    const std::uint64_t rtt_us = (monotonic_ns() - t_now) / 1000ull;
    if (rtt_us > 2000ull || rep.deadline_met == 0) {
      ++misses;
    }
    if (seq % 50 == 0 || rep.alert_tier >= 2) {
      std::printf("[sensor_ingest] seq=%llu rtt=%lluus score_q15=%d tier=%u\n",
                  static_cast<unsigned long long>(seq),
                  static_cast<unsigned long long>(rtt_us),
                  rep.anomaly_score_q15, static_cast<unsigned>(rep.alert_tier));
    }
    next_tick += period_ns;
  }

  ConnectDetach(coid);
  std::printf("[sensor_ingest] Complete: frames=%llu deadline_misses=%llu\n",
              static_cast<unsigned long long>(frames),
              static_cast<unsigned long long>(misses));
  return misses == 0 ? 0 : 2;
#else
  std::uint64_t next_tick = monotonic_ns() + period_ns;
  std::uint64_t alerts = 0;
  for (std::uint64_t seq = 1; seq <= frames; ++seq) {
    sleep_until_ns(next_tick);
    std::int16_t feat[4] = {
        static_cast<std::int16_t>((seq % 60 >= 50) ? 25500 : 5200), 11000, 7000,
        static_cast<std::int16_t>((seq % 60 >= 50) ? 24000 : 4800)};
    const auto score = pulsert::run_q15_mlp_inference(feat);
    if (pulsert::classify_anomaly_q15(score) != pulsert::AlertTier::Nominal)
      ++alerts;
    next_tick += period_ns;
  }
  std::printf("[sensor_ingest_sim] Complete: frames=%llu alerts=%llu\n",
              static_cast<unsigned long long>(frames),
              static_cast<unsigned long long>(alerts));
  return 0;
#endif
}
