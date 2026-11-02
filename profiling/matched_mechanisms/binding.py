"""Independent physical port geometry and byte identity validation."""
from __future__ import annotations


def expected_ports(case):
    h, i, q = case["hidden"], case["intermediate"], case["tokens"]
    if case["family"] == "ffn":
        return {"input": [h, q, 1], "gate_weight": [h, i, 1], "up_weight": [h, i, 1],
                "down_weight": [i, h, 1], "gate": [i, q, 1], "up": [i, q, 1],
                "activated": [i, q, 1], "output": [h, q, 1]}
    d, heads, kvh, length = case["head_dim"], case["heads"], case["kv_heads"], case["history"] + q
    return {"query": [d, q, heads], "new_key": [d, q, kvh], "new_value": [d, q, kvh],
            "key_cache": [d, length, kvh], "value_cache": [d, length, kvh],
            "key_appended": [d, length, kvh], "value_appended": [d, length, kvh],
            "context": [h, q, 1], "hidden": [h, q, 1], "attention_gate": [h, q, 1],
            "output": [h, q, 1], "output_weight": [h, h, 1]}


def validate_ports(case, payload):
    expected = expected_ports(case)
    if set(payload["ports"]) != set(expected) or set(payload["source_values"]) != set(expected):
        raise ValueError("exact physical stage port coverage mismatch")
    for name, shape in expected.items():
        actual = payload["ports"][name]
        if actual["ne"] != shape or actual["bytes"] != shape[0] * shape[1] * shape[2] * 4:
            raise ValueError("physical geometry differs from declared case: " + name)


def fnv_f32_bytes(raw):
    result = 14695981039346656037
    for byte in raw:
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{result:016x}"
