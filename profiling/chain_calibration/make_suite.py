"""Write the predeclared broad shape/split/history grid (no measurements read)."""
from pathlib import Path
import json


def make_suite():
    cases = []
    for hidden, intermediate, tokens, cuts in (
            (128, 256, 1, (32, 128, 224)),
            (256, 512, 1, (64, 256, 448)),
            (256, 512, 8, (64, 256, 448)),
            (512, 1024, 1, (128,)),
            (512, 1024, 32, (128, 512, 896)),
            (2048, 5632, 1, (1408,))):
        for cut in cuts:
            cases.append(dict(id=f"ffn-h{hidden}-i{intermediate}-q{tokens}-cut{cut}",
                              family="ffn", backend="hybrid", dtype="f32", hidden=hidden,
                              intermediate=intermediate, cpu_intermediate=cut, tokens=tokens,
                              history=0, heads=0, kv_heads=0, head_dim=0))
    for hidden, heads, kv_heads, tokens, histories in (
            (128, 4, 1, 1, (0, 128, 1024)),
            (256, 8, 2, 8, (0, 128, 512)),
            (512, 16, 4, 32, (0, 128)),
            (2048, 16, 4, 1, (128,))):
        for history in histories:
            cases.append(dict(id=f"attention-h{hidden}-q{tokens}-hist{history}",
                              family="attention", backend="hybrid", dtype="f32", hidden=hidden,
                              intermediate=0, cpu_intermediate=0, tokens=tokens, history=history,
                              heads=heads, kv_heads=kv_heads, head_dim=hidden//heads))
    return dict(format_version="chain-calibration-suite-0.1", cases=cases,
                protocol=dict(threads=1, cpu_id=0, device=0, warmup=20, samples=51,
                              min_warmup_ms=200, cuda_graphs="disabled"),
                profile_seeds=[1901], heldout_seeds=[2027], repeats=3,
                quality=dict(max_within_process_cv=.15, max_between_process_cv=.20,
                             max_heldout_relative_error=.20),
                bootstrap=dict(iterations=1000, seed=19))


if __name__ == "__main__":
    Path(__file__).with_name("suite.json").write_text(json.dumps(make_suite(), indent=2) + "\n", encoding="utf-8")
