"""Every module in the E/S abstraction package must import on a fresh checkout.

This exists because it stopped being true. When the measurement apparatus and the
study scripts were split out of the repository, `profiling/es_abstraction` was
kept as apparatus while the `experiments.es_abstraction` model it imports was
moved out with the studies -- and nothing failed until the campaign was actually
launched, because no test imported the campaign. An import is the cheapest thing
a clone-and-run repository can check, so it is checked here for the whole
package rather than for the one module that broke.
"""
from __future__ import annotations

import importlib
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "es_abstraction"


def modules():
    names = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        names.append(f"profiling.es_abstraction.{path.stem}")
    return names


class EsAbstractionImports(unittest.TestCase):
    def test_package_is_not_empty(self):
        self.assertTrue(modules(), "no modules discovered under profiling/es_abstraction")

    def test_every_module_imports(self):
        for name in modules():
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_campaign_and_runner_are_reachable(self):
        # The two entry points a campaign is launched through, named explicitly
        # so a rename cannot quietly drop one of them from the sweep above.
        from profiling.es_abstraction import campaign, run_native  # noqa: F401
        self.assertTrue(callable(campaign.main))
        self.assertTrue(callable(run_native.run))


if __name__ == "__main__":
    unittest.main()
