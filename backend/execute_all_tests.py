import sys
import traceback
from pathlib import Path
import tempfile

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.tests.test_data_splitter import (
    test_3way_split_disjoint_indices,
    test_3way_split_reproducibility,
    test_3way_split_stratification_preserves_class_ratios,
    test_3way_split_regression,
    test_3way_split_insufficient_class_samples_fallback,
    test_3way_split_invalid_ratios_and_inputs,
)

from backend.tests.test_baselines import (
    test_classification_baselines_binary,
    test_classification_baselines_multiclass,
    test_regression_baselines,
    test_baselines_zero_data_leakage,
    test_baselines_with_real_turnover_data,
    test_baselines_with_real_house_price_data,
)

from backend.tests.test_reporting import test_decision_summary_and_report_generation
from backend.tests.test_subprocess_runner import (
    test_worker_successful_execution,
    test_worker_stdout_stderr_capture,
    test_worker_crash_handling,
    test_worker_timeout_and_forced_termination,
    test_worker_descendant_process_termination,
    test_worker_memory_limit_termination,
    test_worker_invalid_result_handling,
    test_worker_missing_result_file_handling,
)


def run_all():
    print("=" * 80)
    print("EXECUTING ALL TEST SUITES IN THE REPOSITORY")
    print("=" * 80)

    tests = [
        ("test_data_splitter::test_3way_split_disjoint_indices", test_3way_split_disjoint_indices, False),
        ("test_data_splitter::test_3way_split_reproducibility", test_3way_split_reproducibility, False),
        ("test_data_splitter::test_3way_split_stratification_preserves_class_ratios", test_3way_split_stratification_preserves_class_ratios, False),
        ("test_data_splitter::test_3way_split_regression", test_3way_split_regression, False),
        ("test_data_splitter::test_3way_split_insufficient_class_samples_fallback", test_3way_split_insufficient_class_samples_fallback, False),
        ("test_data_splitter::test_3way_split_invalid_ratios_and_inputs", test_3way_split_invalid_ratios_and_inputs, False),
        ("test_baselines::test_classification_baselines_binary", test_classification_baselines_binary, False),
        ("test_baselines::test_classification_baselines_multiclass", test_classification_baselines_multiclass, False),
        ("test_baselines::test_regression_baselines", test_regression_baselines, False),
        ("test_baselines::test_baselines_zero_data_leakage", test_baselines_zero_data_leakage, False),
        ("test_baselines::test_baselines_with_real_turnover_data", test_baselines_with_real_turnover_data, False),
        ("test_baselines::test_baselines_with_real_house_price_data", test_baselines_with_real_house_price_data, False),
        ("test_reporting::test_decision_summary_and_report_generation", test_decision_summary_and_report_generation, True),
        ("test_subprocess_runner::test_worker_successful_execution", test_worker_successful_execution, True),
        ("test_subprocess_runner::test_worker_stdout_stderr_capture", test_worker_stdout_stderr_capture, True),
        ("test_subprocess_runner::test_worker_crash_handling", test_worker_crash_handling, True),
        ("test_subprocess_runner::test_worker_timeout_and_forced_termination", test_worker_timeout_and_forced_termination, True),
        ("test_subprocess_runner::test_worker_descendant_process_termination", test_worker_descendant_process_termination, True),
        ("test_subprocess_runner::test_worker_memory_limit_termination", test_worker_memory_limit_termination, True),
        ("test_subprocess_runner::test_worker_invalid_result_handling", test_worker_invalid_result_handling, True),
        ("test_subprocess_runner::test_worker_missing_result_file_handling", test_worker_missing_result_file_handling, True),
    ]


    passed = 0
    failed = 0

    for name, test_fn, needs_tmp in tests:
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    test_fn(Path(tmp_dir))
            else:
                test_fn()
            print(f"PASSED [OK]: {name}")
            passed += 1
        except Exception as e:
            print(f"FAILED [X]: {name}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: Total={len(tests)} | Passed={passed} | Failed={failed}")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
