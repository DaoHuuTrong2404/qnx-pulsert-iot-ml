#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace pulsert {

struct RtTaskSpec {
  const char *name;
  std::uint32_t priority; // Higher number = higher SCHED_FIFO priority
  double wcet_us;         // C_i
  double period_us;       // T_i
  double deadline_us;     // D_i <= T_i
  double blocking_us; // B_i (bounded single IPC critical section under QNX PI)
};

struct RtaTaskResult {
  const char *name;
  double response_us;
  double deadline_us;
  bool schedulable;
};

struct RtaSystemReport {
  double total_utilization;
  double liu_layland_bound;
  bool exact_schedulable;
  std::vector<RtaTaskResult> tasks;
};

// Exact Fixed-Point Response-Time Analysis (RTA) for Fixed-Priority SCHED_FIFO
// + QNX Priority Inheritance. Tasks should be independent of order; higher
// priority value preempts lower priority value.
inline RtaSystemReport
analyze_schedulability(const std::vector<RtTaskSpec> &specs) {
  RtaSystemReport report{};
  const std::size_t n = specs.size();
  double util = 0.0;
  for (const auto &t : specs) {
    util += t.wcet_us / t.period_us;
  }
  report.total_utilization = util;
  report.liu_layland_bound =
      n == 0 ? 1.0
             : static_cast<double>(n) *
                   (std::pow(2.0, 1.0 / static_cast<double>(n)) - 1.0);
  report.exact_schedulable = true;

  for (std::size_t i = 0; i < n; ++i) {
    const auto &ti = specs[i];
    double r = ti.wcet_us + ti.blocking_us;
    bool ok = false;
    for (int iter = 0; iter < 512; ++iter) {
      double next_r = ti.wcet_us + ti.blocking_us;
      for (std::size_t j = 0; j < n; ++j) {
        if (i == j)
          continue;
        if (specs[j].priority > ti.priority) {
          next_r += std::ceil(r / specs[j].period_us) * specs[j].wcet_us;
        }
      }
      if (next_r > ti.deadline_us) {
        r = next_r;
        ok = false;
        break;
      }
      if (std::fabs(next_r - r) < 1e-6) {
        r = next_r;
        ok = true;
        break;
      }
      r = next_r;
    }
    if (!ok)
      report.exact_schedulable = false;
    report.tasks.push_back(RtaTaskResult{ti.name, r, ti.deadline_us, ok});
  }
  return report;
}

} // namespace pulsert
