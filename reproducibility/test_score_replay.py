"""Integrity tests for saved-score replay, not material-model experiments."""
import copy
import math
from pathlib import Path
import unittest

from replay_io import load_json
from score_replay import aggregate, validate_marginal


class ScoreIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.rows = load_json(root / "ledger/marginal_scores.json")["rows"]
        cls.formulations = load_json(root / "provenance/SOURCES.json")["formulation_P_ids"]

    def reject(self, rows):
        with self.assertRaises(ValueError):
            validate_marginal(rows, self.formulations)

    def test_missing_case_is_not_silently_dropped(self):
        self.reject(self.rows[:-1])

    def test_duplicate_case_is_not_a_new_formulation(self):
        self.reject(self.rows + [self.rows[0]])

    def test_target_count_change_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["target_count"] = 32
        self.reject(rows)

    def test_nonfinite_loss_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["crps"] = math.nan
        self.reject(rows)

    def test_negative_MSE_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["mean_mse"] = -1
        self.reject(rows)

    def test_coverage_count_must_match_saved_fraction(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["covered_count90"] += 1
        self.reject(rows)

    def test_continuation_score_mass_is_preserved(self):
        rows = copy.deepcopy(self.rows)
        next(r for r in rows if r["mask"] == "one_sided")["continuation_crps_sum"] += 1
        self.reject(rows)

    def test_pooled_RMSE_uses_squared_errors(self):
        # A two-scalar arithmetic example, not simulated measured material data.
        common = {"target_count": 33, "crps": 0, "joint_log_density": 0,
                  "covered_count90": 0, "width90": 0}
        result = aggregate([{**common, "mean_mse": 0, "P": 0}, {**common, "mean_mse": 4, "P": 1}])
        self.assertAlmostEqual(result["RMSE"], math.sqrt(2))
        self.assertNotAlmostEqual(result["RMSE"], 1)


if __name__ == "__main__":
    unittest.main()
