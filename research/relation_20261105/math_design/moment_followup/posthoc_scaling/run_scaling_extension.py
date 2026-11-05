"""Post-hoc synthetic scaling extension; no material or previous-case execution."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

PINNED_SOURCE = {
    'moment_core.py': 'fa2e63d672a23cc490786ca21f03161122e7670d9c9d8457f6a123a710025de8',
    'run_moment_control.py': 'd55f39d9cff4931cb3bd63a7669b804934764a755fc23e5c064bd1a189823a62'}
PINNED_PRIOR_RESULT = 'cc48fe6581545569760d544da6d5a1342ce36a2ab0855a1fb3fad80787c37893'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def configuration():
    return {'version': 'posthoc_moment_scaling_extension_v1',
            'hypotheses': [256, 512, 1024, 2048], 'rmax': [.25, .6, .9, .99],
            'repair_actions': 8, 'proxies': 16, 'outcomes_per_repair': 2,
            'epsilon_CRPS': 1e-6, 'degree_cap': 64, 'main_case_count': 16,
            'timing_repetitions': 1, 'BLAS_threads': 1,
            'case_order': 'H ascending outer loop, rmax ascending inner loop',
            'frozen_function_reused_without_modification': 'run_moment_control.run_case',
            'prior_H_8_64_128_reexecuted': False, 'mathematical_controls_reexecuted': False,
            'minimum_initial_free_physical_bytes': 4294967296,
            'conservative_maximum_temporary_reserve_bytes': 1610612736,
            'largest_cached_kernel_bank_numeric_bytes': 536870912,
            'timing_is_not_end_to_end_policy_latency': True,
            'exploratory_posthoc_scaling_not_independent_material_confirmation': True,
            'outcome_based_case_filtering_or_retuning': False,
            'material_execution': False}


def available_physical_memory():
    """Read actual available Windows physical RAM without loading the producer."""
    if sys.platform != 'win32':
        raise RuntimeError('This bounded environment guard supports Windows only')
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed; no computation is permitted')
    return {'actual_utc': utc(), 'available_physical_bytes': int(status.ullAvailPhys),
            'total_physical_bytes': int(status.ullTotalPhys),
            'source': 'Windows GlobalMemoryStatusEx', 'producer_imported_at_check': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    frozen = folder.parent
    source = {'run_scaling_extension.py': sha(__file__)}
    plan, approval = load(args.plan), load(args.approval)
    keys = {'kind', 'status', 'registered_actual_utc', 'source_sha256',
            'frozen_source_sha256', 'prior_result', 'configuration', 'no_material_execution'}
    if (set(plan) != keys or plan['kind'] != 'posthoc_moment_computational_scaling'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != source
            or plan['frozen_source_sha256'] != PINNED_SOURCE
            or plan['configuration'] != configuration() or plan['no_material_execution'] is not True):
        raise ValueError('Exact registered extension source and configuration required')
    if (approval.get('approved') is not True or approval.get('scope') != 'POSTHOC_MOMENT_SCALING_EXTENSION'
            or approval.get('plan_sha256') != sha(args.plan)):
        raise ValueError('Root approval of this exact extension plan is required')
    if (plan['prior_result']['sha256'] != PINNED_PRIOR_RESULT
            or sha(plan['prior_result']['file']) != PINNED_PRIOR_RESULT):
        raise ValueError('The already-consumed twelve-case result must be pinned unchanged')
    if PINNED_SOURCE != {name: sha(frozen / name) for name in PINNED_SOURCE}:
        raise ValueError('The two reused producer files changed')
    memory = available_physical_memory()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if memory['available_physical_bytes'] < configuration()['minimum_initial_free_physical_bytes']:
        write(output / 'PREEXECUTION_RESOURCE_REFUSAL.json', {'actual_utc': utc(), 'memory': memory,
              'status': 'STOP_BEFORE_PRODUCER_IMPORT', 'minimum_bytes': 4294967296,
              'source_sha256': source, 'plan_sha256': sha(args.plan),
              'no_case_started': True, 'no_automatic_retry_or_case_subset': True})
        raise RuntimeError('Actual free physical RAM is below four GiB; no case started')
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'source_sha256': source,
          'frozen_source_sha256': PINNED_SOURCE, 'prior_result': plan['prior_result'],
          'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
          'memory_before_producer_import': memory,
          'reserved_budget_is_conservative_estimate_not_a_memory_guarantee': True,
          'posthoc_scaling_only': True, 'old_cases_or_mathematical_controls_executed': False})
    try:
        sys.path.insert(0, str(frozen))
        spec = importlib.util.spec_from_file_location('frozen_moment_runner', frozen / 'run_moment_control.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = []
        with module.threadpool_limits(limits=1):
            for h in configuration()['hypotheses']:
                for r in configuration()['rmax']:
                    case = module.run_case(h, r)
                    cases.append(case)
                    name = 'CASE_H' + str(h) + '_r' + str(r).replace('.', 'p') + '.json'
                    write(output / name, module.encoded({'actual_end_utc': utc(), 'case': case}))
                    print(json.dumps({'completed_H': h, 'completed_rmax': r, 'posthoc_scaling_only': True}), flush=True)
        if source != {'run_scaling_extension.py': sha(__file__)} or PINNED_SOURCE != {name: sha(frozen / name) for name in PINNED_SOURCE}:
            raise ValueError('A bound source changed during the approved extension')
        write(output / 'RESULTS.json', module.encoded({'status': 'COMPLETE_POSTHOC_MOMENT_SCALING',
            'actual_end_utc': utc(), 'source_sha256': source, 'frozen_source_sha256': PINNED_SOURCE,
            'prior_result': plan['prior_result'], 'configuration': configuration(), 'cases': cases,
            'material_fit_prediction_score_calls': 0,
            'scope': 'Exploratory computational scaling, including losing points; neither a new scientific effect nor independent material confirmation.',
            'one_ordered_timing_not_end_to_end_policy_latency': True}))
        print(json.dumps({'status': 'COMPLETE_POSTHOC_MOMENT_SCALING', 'case_count': len(cases)}), flush=True)
    except Exception as error:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
            'message': str(error), 'retain_all_completed_case_files': True,
            'no_automatic_retry_or_retuning': True})
        raise


if __name__ == '__main__':
    main()
