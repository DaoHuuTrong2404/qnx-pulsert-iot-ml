"""Dual verification for QNX-PulseRT: Fixed-Point Q15 TinyML accuracy + Preemptive SCHED_FIFO RTA vs Discrete PI Simulator."""

from __future__ import annotations

import math

# Exact weights from src/qnx/telemetry_protocol.hpp
W1 = (
    (11469, 8192, 4915, 9830),
    (-6554, 13107, 6554, 3277),
    (9830, -4915, 11469, 8192),
    (4915, 9830, -3277, 13107),
    (-11469, -6554, -4915, -3277),
    (8192, 8192, 8192, 8192),
)
B1 = (-4096, -3277, -3604, -4915, 2458, -5734)
W2 = (10650, 9830, 8192, 11469, -6554, 12288)
B2 = -3277


def c_div_rshift(val: int, bits: int) -> int:
    return val >> bits


def run_q15_mlp(x_q15: tuple[int, int, int, int]) -> int:
    hidden = []
    for h in range(6):
        acc = B1[h]
        for i in range(4):
            prod = x_q15[i] * W1[h][i]
            assert -2147483648 <= prod <= 2147483647, "int32 MAC overflow"
            acc += c_div_rshift(prod, 15)
        hidden.append(min(32767, acc) if acc > 0 else 0)

    out = B2
    for h in range(6):
        prod = hidden[h] * W2[h]
        assert -2147483648 <= prod <= 2147483647, "int32 output MAC overflow"
        out += c_div_rshift(prod, 14)

    res = 16384 + c_div_rshift(out, 1)
    return max(0, min(32767, res))


def run_float_ref_mlp(x_q15: tuple[int, int, int, int]) -> float:
    xf = [v / 32768.0 for v in x_q15]
    hidden = []
    for h in range(6):
        acc = B1[h] / 16384.0
        for i in range(4):
            acc += xf[i] * (W1[h][i] / 16384.0)
        hidden.append(min(32767.0 / 16384.0, max(0.0, acc)))
    out = B2 / 16384.0
    for h in range(6):
        out += hidden[h] * (W2[h] / 16384.0)
    return max(0.0, min(32767.0 / 32768.0, 0.5 + out / 4.0))


def rta_analytical(tasks: list[dict]) -> dict:
    n = len(tasks)
    util = sum(t["wcet"] / t["period"] for t in tasks)
    ll_bound = n * (2.0 ** (1.0 / n) - 1.0)
    results = []
    all_ok = True

    for i, ti in enumerate(tasks):
        r = ti["wcet"] + ti["blocking"]
        ok = False
        for _ in range(512):
            nxt = ti["wcet"] + ti["blocking"]
            for j, tj in enumerate(tasks):
                if i != j and tj["priority"] > ti["priority"]:
                    nxt += math.ceil(r / tj["period"]) * tj["wcet"]
            if nxt > ti["deadline"]:
                r = nxt
                ok = False
                break
            if abs(nxt - r) < 1e-6:
                r = nxt
                ok = True
                break
            r = nxt
        if not ok:
            all_ok = False
        results.append(
            {
                "name": ti["name"],
                "response": r,
                "deadline": ti["deadline"],
                "schedulable": ok,
            }
        )

    return {
        "util": util,
        "ll_bound": ll_bound,
        "exact_schedulable": all_ok,
        "tasks": results,
    }


def simulate_fifo_hyperperiod(
    tasks: list[dict], horizon: int, shared_cs_us: int = 45
) -> dict:
    """Discrete 1-us preemptive SCHED_FIFO + QNX priority-inheritance mutex simulator."""
    rem = [0] * len(tasks)
    release_t = [0] * len(tasks)
    max_resp = [0] * len(tasks)
    misses = [0] * len(tasks)

    low_idx = min(range(len(tasks)), key=lambda i: tasks[i]["priority"])
    cs_rem = shared_cs_us

    for t in range(horizon):
        for idx, spec in enumerate(tasks):
            if t % spec["period"] == 0:
                if rem[idx] > 0:
                    misses[idx] += 1
                rem[idx] += spec["wcet"]
                release_t[idx] = t

        if cs_rem > 0:
            # Low-priority task holds shared IPC mutex at critical instant -> inherits top priority
            active = low_idx
            cs_rem -= 1
            if rem[active] > 0:
                rem[active] -= 1
        else:
            ready = [idx for idx in range(len(tasks)) if rem[idx] > 0]
            active = (
                max(ready, key=lambda idx: tasks[idx]["priority"]) if ready else None
            )
            if active is not None:
                rem[active] -= 1

        if active is not None and rem[active] == 0:
            resp = (t + 1) - release_t[active]
            if resp > max_resp[active]:
                max_resp[active] = resp
            if resp > tasks[active]["deadline"]:
                misses[active] += 1

    return {"max_resp": max_resp, "misses": misses}


def main() -> None:
    max_err = 0.0
    nominal_hits = 0
    fault_hits = 0
    for k in range(500):
        if k < 250:
            v = int((k / 250.0) * 14000) - 2000
            x = (v, v // 2, v // 3, v // 2)
        else:
            v = 18000 + ((k - 250) * 57) % 14000
            x = (v, min(32767, v // 2 + 6000), min(32767, v // 3 + 4000), v)
        q15 = run_q15_mlp(x)
        f_q15 = q15 / 32768.0
        f_ref = run_float_ref_mlp(x)
        err = abs(f_q15 - f_ref)
        if err > max_err:
            max_err = err
        if k < 150 and q15 < 17695:
            nominal_hits += 1
        if k >= 400 and q15 >= 24576:
            fault_hits += 1

    assert max_err < 0.003, f"Q15 quantization drift too high: {max_err}"
    assert nominal_hits >= 140, f"Unexpected nominal false positives: {nominal_hits}"
    assert fault_hits >= 95, f"Fault detection sensitivity too low: {fault_hits}"

    nominal_tasks = [
        {
            "name": "sensor_ingest_fifo",
            "priority": 40,
            "wcet": 220,
            "period": 2000,
            "deadline": 2000,
            "blocking": 45,
        },
        {
            "name": "qnx_ml_infer_srv",
            "priority": 30,
            "wcet": 380,
            "period": 2000,
            "deadline": 2000,
            "blocking": 45,
        },
        {
            "name": "aps_telemetry_log",
            "priority": 12,
            "wcet": 950,
            "period": 10000,
            "deadline": 10000,
            "blocking": 0,
        },
    ]
    rta = rta_analytical(nominal_tasks)
    assert rta["exact_schedulable"] is True
    assert rta["util"] < rta["ll_bound"]
    sim = simulate_fifo_hyperperiod(nominal_tasks, 20000, shared_cs_us=45)
    assert sum(sim["misses"]) == 0, f"Nominal schedule had deadline misses: {sim}"
    for idx, t_res in enumerate(rta["tasks"]):
        assert sim["max_resp"][idx] <= t_res["response"] + 1e-6, (
            f"Simulated R ({sim['max_resp'][idx]}) exceeded RTA ({t_res['response']})"
        )

    storm_tasks = [
        {
            "name": "sensor_ingest_fifo",
            "priority": 40,
            "wcet": 700,
            "period": 2000,
            "deadline": 2000,
            "blocking": 100,
        },
        {
            "name": "qnx_ml_infer_srv",
            "priority": 30,
            "wcet": 950,
            "period": 2000,
            "deadline": 2000,
            "blocking": 100,
        },
        {
            "name": "aps_telemetry_log",
            "priority": 12,
            "wcet": 2600,
            "period": 5000,
            "deadline": 5000,
            "blocking": 0,
        },
    ]
    rta_storm = rta_analytical(storm_tasks)
    sim_storm = simulate_fifo_hyperperiod(storm_tasks, 10000, shared_cs_us=100)
    assert rta_storm["exact_schedulable"] is False
    assert sum(sim_storm["misses"]) > 0

    print(
        f"QNX_VERIFIED_OK | q15_max_abs_err={max_err:.6f} | nominal_util={rta['util']:.4f} "
        f"| ll_bound={rta['ll_bound']:.4f} | nominal_R={[round(x['response'], 1) for x in rta['tasks']]} "
        f"| sim_R={sim['max_resp']} | storm_misses={sum(sim_storm['misses'])}"
    )


if __name__ == "__main__":
    main()
