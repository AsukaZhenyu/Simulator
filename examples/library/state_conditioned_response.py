"""Illustrative declared parameters; no hardware accuracy or saved profile required.

Run with the installed library. Measurement/estimation belongs to optional
profiling tools, while the same modeling evaluator owns both predictions here.
"""
from llm_simulator.performance import (
    RequestResponseDelays, ResponseCell, ResponsePoint, ResponseState,
    ResponseTable, ResponseTrajectory, evaluate_response,
)


def main():
    idle = ResponseState(0, 0, 0, 0)
    gpu_work_requested = ResponseState(0, 1, 0, 0)
    evidence = ("illustrative declared durations, not hardware measurements",)
    table = ResponseTable("0" * 64, ("action",), (
        ResponseCell("action", idle, 100, evidence, False),
        ResponseCell("action", gpu_work_requested, 200, evidence, False),
    ))
    requests = ResponseTrajectory(idle, (ResponsePoint(50, gpu_work_requested),))
    immediate = evaluate_response(table, requests)
    delays = RequestResponseDelays(0, 0, 40, 0, evidence, False)
    delayed = evaluate_response(table, requests, request_delays=delays,
                                initial_effective_state=idle)
    print("Declared example, uncalibrated; durations in ns")
    print("Immediate request response:", immediate.duration_ns)  # 150
    print("40 ns GPU worker activation delay:", delayed.duration_ns)  # 110
    print("Both predictions retain completed work and use the common event engine.")


if __name__ == "__main__":
    main()
