from __future__ import annotations

import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any


def expected_graph_fragments(layer_count: int, attention_layer_count: int) -> int:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if attention_layer_count < 0 or attention_layer_count > layer_count:
        raise ValueError("attention_layer_count must be between zero and layer_count")
    return 3 * layer_count + attention_layer_count + 1


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return sorted(values)[index]


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [str(column[0]) for column in cursor.description or ()]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _covered_nanoseconds(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    covered = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    return covered + current_end - current_start


def analyze_nsys_sqlite(
    sqlite_path: str | Path,
    *,
    generation_tokens: int,
    layer_count: int | None = None,
    attention_layer_count: int | None = None,
    ssm_layer_count: int | None = None,
    embedding_bytes: int | None = None,
) -> dict[str, Any]:
    """Extract a steady-state decode decomposition from an Nsight SQLite export.

    The first token is excluded because it captures/instantiates CUDA graphs. The
    final token is excluded because the next token start is unavailable. Token
    starts are inferred from evenly sized groups of cudaGraphLaunch calls. When
    architecture metadata is available, the final ``generation_tokens`` groups
    are selected so a preceding cached-depth prefill can share the same trace.
    """
    source = Path(sqlite_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if generation_tokens < 3:
        raise ValueError("generation_tokens must be at least three")
    if layer_count is not None and layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if ssm_layer_count is not None and ssm_layer_count < 0:
        raise ValueError("ssm_layer_count cannot be negative")
    if embedding_bytes is not None and embedding_bytes <= 0:
        raise ValueError("embedding_bytes must be positive")

    with sqlite3.connect(source) as connection:
        launches = connection.execute(
            """
            SELECT runtime.start, runtime.end
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON strings.id = runtime.nameId
            WHERE strings.value LIKE 'cudaGraphLaunch%'
            ORDER BY runtime.start
            """
        ).fetchall()
        if not launches:
            raise ValueError("Nsight export contains no cudaGraphLaunch calls")
        expected_fragments = None
        if layer_count is not None and attention_layer_count is not None:
            expected_fragments = expected_graph_fragments(
                layer_count, attention_layer_count
            )
        total_graph_launches = len(launches)
        prefix_graph_launches = 0
        if expected_fragments is not None:
            decode_graph_launches = generation_tokens * expected_fragments
            if total_graph_launches < decode_graph_launches:
                raise ValueError(
                    "Nsight export has fewer cudaGraphLaunch calls than the "
                    "requested decode suffix: "
                    f"{total_graph_launches} vs {decode_graph_launches}"
                )
            prefix_graph_launches = total_graph_launches - decode_graph_launches
            launches = launches[-decode_graph_launches:]
            graphs_per_token = expected_fragments
        else:
            if total_graph_launches % generation_tokens != 0:
                raise ValueError(
                    "cudaGraphLaunch count is not divisible by generation_tokens: "
                    f"{total_graph_launches} vs {generation_tokens}"
                )
            graphs_per_token = total_graph_launches // generation_tokens
        token_starts = [
            int(launches[token * graphs_per_token][0])
            for token in range(generation_tokens)
        ]
        token_seconds = [
            (token_starts[index + 1] - token_starts[index]) / 1e9
            for index in range(1, generation_tokens - 1)
        ]
        stable_token_count = len(token_seconds)
        stable_start = token_starts[1]
        stable_end = token_starts[-1]

        api_rows = _rows(
            connection.execute(
                """
                SELECT strings.value AS api,
                       COUNT(*) AS calls,
                       SUM(runtime.end - runtime.start) AS total_ns,
                       AVG(runtime.end - runtime.start) AS average_ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
                JOIN StringIds AS strings ON strings.id = runtime.nameId
                WHERE runtime.start >= ? AND runtime.start < ?
                GROUP BY strings.value
                ORDER BY total_ns DESC
                """,
                (stable_start, stable_end),
            )
        )
        for row in api_rows:
            row["calls_per_token"] = int(row["calls"]) / stable_token_count
            row["milliseconds_per_token"] = (
                int(row["total_ns"]) / stable_token_count / 1e6
            )

        copy_rows = _rows(
            connection.execute(
                """
                SELECT kinds.label AS direction,
                       COUNT(*) AS copies,
                       SUM(memcpy.bytes) AS bytes,
                       SUM(memcpy.end - memcpy.start) AS total_ns
                FROM CUPTI_ACTIVITY_KIND_MEMCPY AS memcpy
                JOIN ENUM_CUDA_MEMCPY_OPER AS kinds ON kinds.id = memcpy.copyKind
                WHERE memcpy.start >= ? AND memcpy.start < ?
                GROUP BY kinds.label
                ORDER BY bytes DESC
                """,
                (stable_start, stable_end),
            )
        )
        for row in copy_rows:
            row["copies_per_token"] = int(row["copies"]) / stable_token_count
            row["bytes_per_token"] = int(row["bytes"]) / stable_token_count
            row["gpu_milliseconds_per_token"] = (
                int(row["total_ns"]) / stable_token_count / 1e6
            )

        copy_sizes = _rows(
            connection.execute(
                """
                SELECT kinds.label AS direction,
                       memcpy.bytes AS copy_bytes,
                       COUNT(*) AS copies,
                       SUM(memcpy.end - memcpy.start) AS total_ns
                FROM CUPTI_ACTIVITY_KIND_MEMCPY AS memcpy
                JOIN ENUM_CUDA_MEMCPY_OPER AS kinds ON kinds.id = memcpy.copyKind
                WHERE memcpy.start >= ? AND memcpy.start < ?
                GROUP BY kinds.label, memcpy.bytes
                ORDER BY copy_bytes DESC, copies DESC
                """,
                (stable_start, stable_end),
            )
        )
        for row in copy_sizes:
            row["copies_per_token"] = int(row["copies"]) / stable_token_count
            row["gpu_milliseconds_per_token"] = (
                int(row["total_ns"]) / stable_token_count / 1e6
            )

        kernel_row = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(end - start), 0)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start >= ? AND start < ?
            """,
            (stable_start, stable_end),
        ).fetchone()
        gpu_intervals = [
            (max(stable_start, int(start)), min(stable_end, int(end)))
            for start, end in connection.execute(
                """
                SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE start < ? AND end > ?
                UNION ALL
                SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE start < ? AND end > ?
                """,
                (stable_end, stable_start, stable_end, stable_start),
            ).fetchall()
        ]

    direction_rows = {str(row["direction"]): row for row in copy_rows}
    d2h_bytes = float(direction_rows.get("Device-to-Host", {}).get("bytes_per_token", 0))
    h2d_bytes = float(direction_rows.get("Host-to-Device", {}).get("bytes_per_token", 0))
    embedding_copies = 0
    embedding_gpu_ns = 0
    if embedding_bytes is not None:
        matching = [
            row
            for row in copy_sizes
            if row["direction"] == "Device-to-Host"
            and int(row["copy_bytes"]) == embedding_bytes
        ]
        if matching:
            embedding_copies = int(matching[0]["copies"])
            embedding_gpu_ns = int(matching[0]["total_ns"])

    gpu_active_ns = _covered_nanoseconds(gpu_intervals)
    steady_elapsed_ns = stable_end - stable_start
    gpu_idle_or_untraced_ns = max(0, steady_elapsed_ns - gpu_active_ns)

    return {
        "schema_version": 2,
        "source_sqlite": source.as_posix(),
        "generation_tokens": generation_tokens,
        "steady_state": {
            "excluded_tokens": [1, generation_tokens],
            "token_count": stable_token_count,
            "mean_token_seconds": statistics.fmean(token_seconds),
            "median_token_seconds": statistics.median(token_seconds),
            "p95_token_seconds": _percentile(token_seconds, 0.95),
            "min_token_seconds": min(token_seconds),
            "max_token_seconds": max(token_seconds),
            "tokens_per_second_from_mean": 1.0 / statistics.fmean(token_seconds),
        },
        "graph_fragmentation": {
            "total_graph_launches": total_graph_launches,
            "selected_decode_graph_launches": len(launches),
            "excluded_prefix_graph_launches": prefix_graph_launches,
            "graphs_per_token": graphs_per_token,
            "expected_formula": (
                "3 * layer_count + attention_layer_count + 1"
                if expected_fragments is not None
                else None
            ),
            "expected_graphs_per_token": expected_fragments,
            "matches_expected_formula": (
                graphs_per_token == expected_fragments
                if expected_fragments is not None
                else None
            ),
        },
        "architecture": {
            "layer_count": layer_count,
            "attention_layer_count": attention_layer_count,
            "ssm_layer_count": ssm_layer_count,
        },
        "embedding_fallback": {
            "expected_token_embedding_bytes": embedding_bytes,
            "d2h_copies": embedding_copies,
            "d2h_copies_per_token": embedding_copies / stable_token_count,
            "d2h_gpu_milliseconds_per_token": (
                embedding_gpu_ns / stable_token_count / 1e6
            ),
            "matches_once_per_token": embedding_copies == stable_token_count,
        },
        "copy_summary": {
            "d2h_bytes_per_token": d2h_bytes,
            "h2d_bytes_per_token": h2d_bytes,
            "non_embedding_d2h_bytes_per_token": max(
                0.0, d2h_bytes - float(embedding_bytes or 0)
            ),
            "directions": copy_rows,
            "sizes": copy_sizes,
        },
        "cuda_api_summary": api_rows,
        "kernel_summary": {
            "kernels": int(kernel_row[0]),
            "kernels_per_token": int(kernel_row[0]) / stable_token_count,
            "gpu_milliseconds_per_token": int(kernel_row[1])
            / stable_token_count
            / 1e6,
        },
        "gpu_timeline": {
            "active_kernel_or_memcpy_milliseconds_per_token": gpu_active_ns
            / stable_token_count
            / 1e6,
            "idle_or_untraced_milliseconds_per_token": gpu_idle_or_untraced_ns
            / stable_token_count
            / 1e6,
            "active_fraction": gpu_active_ns / steady_elapsed_ns,
        },
        "method": [
            "Token starts are inferred from equal-sized cudaGraphLaunch groups.",
            (
                "The final generation-token graph groups are selected; earlier "
                "launches may belong to cached-depth prefill."
                if expected_fragments is not None
                else "The full trace is divided into generation-token graph groups."
            ),
            "Token 1 is excluded because CUDA graphs are captured/instantiated there.",
            "The final token is excluded because a following token start is unavailable.",
            "CUDA API duration sums are service-time diagnostics and can overlap across host threads.",
        ],
    }
