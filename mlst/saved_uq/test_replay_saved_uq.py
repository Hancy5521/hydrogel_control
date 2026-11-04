"""Synthetic guard and weighting fixtures, with hand-checkable expectations.

These fixtures contain made-up score scalars, never material response arrays.
"""
import copy
import unittest
from replay_saved_uq import ALL_METHODS, METHODS, summarize, validate_rows

SOURCE = '10.17638/datacat.liverpool.ac.uk/1296'


def fixture():
    rows = []
    # Group A has two scalar records (losses 1,3), group B one (loss 8).
    # Correct group-macro loss is (2+8)/2=5, not naive curve mean 4.
    for curve, group, loss, coverage in [('a1','A',1.,0.), ('a2','A',3.,.5), ('b','B',8.,1.)]:
        for method in ALL_METHODS:
            s = {'n_targets': 2, 'mse_log': loss, 'rmse_log': loss**.5, 'mae_log': loss**.5}
            if method in METHODS:
                s.update(crps_log=loss, coverage90=coverage, whole_curve_covered90=(coverage == 1.),
                         width90_log=1., interval_score90_log=loss+1., marginal_nll_log=loss-2.)
            rows.append({'curve_id':curve,'group_id':group,'source_doi':SOURCE,'method':method,'scores':s})
    return rows


class SavedScalarGuards(unittest.TestCase):
    def test_hand_weighted_group_expectation(self):
        records = summarize(fixture(), expected_cohort=False)
        self.assertEqual(len(records), 3)
        for r in records:
            self.assertEqual(r['crps_log'], 5.)
            self.assertEqual(r['root_group_macro_mse_log'], 5**.5)
            self.assertEqual(r['coverage90'], .625)
            self.assertEqual(r['zero_coverage_native_curves'], 1)
            self.assertEqual(r['all_targets_in_marginal_bands_native_curves'], 1)

    def test_duplicate_row_rejected(self):
        r = fixture(); r.append(copy.deepcopy(r[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_rows(r, expected_cohort=False)

    def test_missing_method_rejected(self):
        r = fixture(); r.pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete method'):
            validate_rows(r, expected_cohort=False)

    def test_cross_method_identity_mismatch_rejected(self):
        r = fixture(); r[0]['group_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Methods disagree'):
            validate_rows(r, expected_cohort=False)

    def test_wrong_source_rejected(self):
        r = fixture(); r[0]['source_doi'] = 'unregistered'
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            validate_rows(r, expected_cohort=False)

    def test_nonfinite_rejected(self):
        r = fixture(); r[0]['scores']['mse_log'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            validate_rows(r, expected_cohort=False)

    def test_array_rejected(self):
        r = fixture(); r[0]['scores']['test_array'] = [1,2]
        with self.assertRaisesRegex(ValueError, 'Nonscalar'):
            validate_rows(r, expected_cohort=False)

    def test_probability_indicator_mismatch_rejected(self):
        r = fixture()
        one = next(s for s in r if s['method'] in METHODS)
        one['scores']['whole_curve_covered90'] = True
        with self.assertRaisesRegex(ValueError, 'indicator disagrees'):
            validate_rows(r, expected_cohort=False)

    def test_small_fixture_cannot_impersonate_frozen_cohort(self):
        with self.assertRaisesRegex(ValueError, 'Expected original'):
            validate_rows(fixture())


if __name__ == '__main__':
    unittest.main(verbosity=2)
