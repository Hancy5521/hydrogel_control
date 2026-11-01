"""Integrity tests for saved-score replay, not material-model experiments."""
import copy
import math
from pathlib import Path
import unittest

from replay_io import load_json
from score_replay import aggregate, make_tables, validate_marginal


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

    def test_S7_complete_methods_and_pooled_RMSE(self):
        root = Path(__file__).resolve().parent
        variogram = load_json(root / "ledger/variogram_scores.json")["rows"]
        points = load_json(root / "ledger/point_only_scores.json")["rows"]
        facts = load_json(root / "ledger/reported_facts.json")
        provenance = load_json(root / "provenance/SOURCES.json")
        result = make_tables(self.rows, variogram, points, facts, provenance)
        s7 = result["tables"]["SI_tableS7"]
        expected_labels = ["Coordinate GP", "Generic bounded tilt", "Curvature bounded tilt",
                           "Conditional Gaussian", "Gaussian curvature correction",
                           "Matched random Gaussian", "Bayesian Cross", "Bayesian Carreau a2",
                           "Fixed Student-t diagnostic", "PCHIP endpoint cubic"]
        self.assertEqual([r["model"] for r in s7], expected_labels)
        expected_cases = {(p, layout) for p in self.formulations
                          for layout in ("spread", "middle_gap", "one_sided")}
        self.assertEqual(len(expected_cases), 75)
        for model, row in zip(provenance["models"], s7[:-1]):
            cases = [(r["P"], r["mask"]) for r in self.rows if r["model"] == model]
            self.assertEqual(len(cases), 75)
            self.assertEqual(set(cases), expected_cases)
            self.assertAlmostEqual(row["pooled_RMSE"], facts["aggregate"][model]["RMSE"], delta=2e-11)
            for layout in ("spread", "middle_gap", "one_sided"):
                self.assertAlmostEqual(row[layout + "_RMSE"], facts["by_mask"][layout][model]["RMSE"], delta=2e-11)
        point = s7[-1]
        self.assertEqual({(r["P"], r["mask"]) for r in points}, expected_cases)
        self.assertEqual(set(point), {"model", "spread_RMSE", "middle_gap_RMSE", "one_sided_RMSE",
                                      "pooled_RMSE", "prediction_role"})
        self.assertEqual(point["prediction_role"], "Point-only; no probability law assigned")
        self.assertAlmostEqual(point["pooled_RMSE"], facts["PCHIP_RMSE"], delta=2e-11)
        for layout in ("spread", "middle_gap", "one_sided"):
            self.assertAlmostEqual(point[layout + "_RMSE"], facts["PCHIP_by_mask_RMSE"][layout], delta=2e-11)
        # The actual saved point errors are heterogeneous: averaging query
        # RMSEs would give a materially different answer to pooled RMSE.
        wrong_mean_RMSE = math.fsum(math.sqrt(r["mean_mse"]) for r in points) / len(points)
        self.assertNotAlmostEqual(point["pooled_RMSE"], wrong_mean_RMSE, places=5)
        with self.assertRaises(ValueError):
            make_tables(self.rows, variogram, points[:-1], facts, provenance)


if __name__ == "__main__":
    unittest.main()
