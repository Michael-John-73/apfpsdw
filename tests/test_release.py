"""Focused release and numerical regression tests; not additional experiments."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_B" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from scipy.stats import betabinom
from paper_b.probability import acceptance, bb_pmf, temperature_pmf
from paper_b.config import load
from verify_release import paired_bounds, verify_native, verify_prediction_summary


class ReleaseTests(unittest.TestCase):
    def test_native_recount(self):
        self.assertEqual(verify_native(), {"rows": 18000, "cells": 18, "contrasts": 6})

    def test_prediction_limits(self):
        verify_prediction_summary()

    def test_beta_binomial_matches_scipy(self):
        for length in (48, 136):
            np.testing.assert_allclose(
                bb_pmf(length, 2.5, 6),
                betabinom.pmf(np.arange(length+1), length, 2.5, 6),
                atol=1e-13,
            )

    def test_temperature_and_acceptance(self):
        pmf = bb_pmf(48, 2, 8)
        calibrated = temperature_pmf(pmf, .75)
        self.assertAlmostEqual(float(calibrated.sum()), 1)
        decisions = acceptance(calibrated)
        self.assertEqual(decisions["p_exact"], calibrated[0])
        self.assertAlmostEqual(decisions["p_tau"], float(calibrated[:7].sum()))

    def test_native_lower_bound(self):
        low, _ = paired_bounds(783, 0, 1000)
        self.assertAlmostEqual(low, .7478817373089898)
        with self.assertRaises(ValueError):
            paired_bounds(900, 200, 1000)

    def test_portable_config(self):
        config = load(ROOT / "configs/component.template.json")
        self.assertEqual(config["study_id"], "new_component_study")
        self.assertEqual(config["data"]["fresh_sources"], [])


if __name__ == "__main__":
    unittest.main()
