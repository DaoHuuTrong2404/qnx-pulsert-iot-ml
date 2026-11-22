#pragma once

#include <cstdint>
#include <type_traits>

namespace pulsert {

constexpr std::uint32_t kProtocolMagic = 0x51505254u; // "QPRT"
constexpr std::uint16_t kMsgTypeSensorFrame = 1;
constexpr std::uint16_t kMsgTypeHealthQuery = 2;

enum class AlertTier : std::uint8_t {
  Nominal = 0,
  Warning = 1,
  CriticalTrip = 2
};

// Zero-heap fixed-layout IPC message transferred synchronously over QNX
// MsgSend/MsgReceive
struct SensorFrameMsg {
  std::uint32_t magic;
  std::uint16_t msg_type;
  std::uint16_t sensor_id;
  std::uint64_t seq;
  std::uint64_t t_capture_ns;
  // Normalized Q15 fixed-point features [-32768..32767 representing
  // -1.0..~1.0]: [0]=vibration_rms, [1]=bearing_temp_norm,
  // [2]=phase_current_norm, [3]=acoustic_kurtosis
  std::int16_t features_q15[4];
};

struct InferenceReplyMsg {
  std::uint32_t magic;
  std::uint64_t seq;
  std::uint64_t t_reply_ns;
  std::uint32_t exec_us;
  std::int32_t anomaly_score_q15; // [0..32767] -> [0.0..1.0]
  std::uint8_t alert_tier;
  std::uint8_t deadline_met;
  std::uint8_t reserved[2];
};

static_assert(std::is_trivially_copyable_v<SensorFrameMsg>,
              "IPC msg must be trivially copyable");
static_assert(std::is_trivially_copyable_v<InferenceReplyMsg>,
              "IPC reply must be trivially copyable");

inline std::int32_t clamp_i32(std::int32_t v, std::int32_t lo,
                              std::int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Deterministic Q15 piecewise-linear hard-sigmoid / ReLU TinyML MLP:
// Architecture: 4 -> 6 (ReLU) -> 1 (clamped sigmoid Q15)
// Trained weights quantized to Q14 [-16384..16384] for zero-overflow int32
// MACs.
struct TinyMlWeights {
  static constexpr std::int16_t W1[6][4] = {
      {11469, 8192, 4915, 9830},     {-6554, 13107, 6554, 3277},
      {9830, -4915, 11469, 8192},    {4915, 9830, -3277, 13107},
      {-11469, -6554, -4915, -3277}, {8192, 8192, 8192, 8192}};
  static constexpr std::int32_t B1[6] = {-4096, -3277, -3604,
                                         -4915, 2458,  -5734};
  static constexpr std::int16_t W2[6] = {10650, 9830,  8192,
                                         11469, -6554, 12288};
  static constexpr std::int32_t B2 = -3277;
};

// Pure fixed-point inference with O(1) deterministic WCET and zero dynamic
// allocation.
inline std::int32_t run_q15_mlp_inference(const std::int16_t x_q15[4]) {
  std::int32_t hidden[6];
  for (int h = 0; h < 6; ++h) {
    std::int32_t acc =
        TinyMlWeights::B1[h]; // Q14 bias aligned with (Q15 * Q14) >> 15
    for (int i = 0; i < 4; ++i) {
      acc += (static_cast<std::int32_t>(x_q15[i]) *
              static_cast<std::int32_t>(TinyMlWeights::W1[h][i])) >>
             15;
    }
    // ReLU in Q14 clamped to [0, 32767]
    hidden[h] = acc > 0 ? (acc > 32767 ? 32767 : acc) : 0;
  }

  std::int32_t out = TinyMlWeights::B2;
  for (int h = 0; h < 6; ++h) {
    out += (hidden[h] * static_cast<std::int32_t>(TinyMlWeights::W2[h])) >> 14;
  }
  // Map Q14 logit [-16384..16384] via hard-sigmoid to Q15 [0..32767]
  // p = clamp(0.5 + out / 4.0, 0, 1) -> in Q15: 16384 + (out >> 1)
  return clamp_i32(16384 + (out >> 1), 0, 32767);
}

inline AlertTier classify_anomaly_q15(std::int32_t score_q15) {
  if (score_q15 >= 24576)
    return AlertTier::CriticalTrip; // >= 0.75
  if (score_q15 >= 17695)
    return AlertTier::Warning; // >= 0.54
  return AlertTier::Nominal;
}

} // namespace pulsert
