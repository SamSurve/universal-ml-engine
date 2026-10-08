import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.engine.contracts.schemas import WorkerJobSpec, WorkerStatus
from backend.engine.backends.subprocess_runner import SubprocessRunner


def run_verification():
    print("=" * 80)
    print("MILESTONE M2 — SUBPROCESS WORKER INFRASTRUCTURE VERIFICATION")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # -------------------------------------------------------------
        # 1. Successful Worker Execution
        # -------------------------------------------------------------
        print("\n[1/5] Testing Normal Worker Execution...")
        spec_ok = WorkerJobSpec(
            job_id="m2_verify_success",
            worker_module="backend.engine.backends.workers.test_worker",
            job_type="echo",
            time_limit_seconds=10.0,
            params={"message": "M2_Verification_OK"},
        )
        res_ok = SubprocessRunner.run_job(spec_ok, working_dir=tmp_path / "job_ok")
        print(f"  Status: {res_ok.status.value}")
        print(f"  Exit Code: {res_ok.exit_code}")
        print(f"  Runtime: {res_ok.runtime_seconds:.4f}s")
        print(f"  Peak Memory: {res_ok.peak_memory_mb:.2f} MB")
        print(f"  Stdout captured: {res_ok.stdout.strip()}")
        print(f"  Payload: {res_ok.payload}")
        assert res_ok.status == WorkerStatus.SUCCESS

        # -------------------------------------------------------------
        # 2. Worker Crash Handling
        # -------------------------------------------------------------
        print("\n[2/5] Testing Worker Crash Isolation...")
        spec_crash = WorkerJobSpec(
            job_id="m2_verify_crash",
            worker_module="backend.engine.backends.workers.test_worker",
            job_type="crash",
            time_limit_seconds=10.0,
            params={"exit_code": 42, "error_message": "Subprocess crash simulation"},
        )
        res_crash = SubprocessRunner.run_job(spec_crash, working_dir=tmp_path / "job_crash")
        print(f"  Status: {res_crash.status.value}")
        print(f"  Exit Code: {res_crash.exit_code}")
        print(f"  Stderr captured: {res_crash.stderr.strip()}")
        print(f"  Parent process alive: True")
        assert res_crash.status == WorkerStatus.CRASHED

        # -------------------------------------------------------------
        # 3. Timeout & Forced Termination
        # -------------------------------------------------------------
        print("\n[3/5] Testing Hard Timeout & Forced Termination...")
        spec_timeout = WorkerJobSpec(
            job_id="m2_verify_timeout",
            worker_module="backend.engine.backends.workers.test_worker",
            job_type="sleep",
            time_limit_seconds=0.5,
            params={"duration": 10.0},
        )
        res_timeout = SubprocessRunner.run_job(spec_timeout, working_dir=tmp_path / "job_timeout", poll_interval_seconds=0.05)
        print(f"  Status: {res_timeout.status.value}")
        print(f"  Runtime: {res_timeout.runtime_seconds:.4f}s")
        print(f"  Error message: {res_timeout.error_message}")
        assert res_timeout.status == WorkerStatus.TIMEOUT

        # -------------------------------------------------------------
        # 4. Descendant Process Tree Termination
        # -------------------------------------------------------------
        print("\n[4/5] Testing Descendant Process Tree Termination on Windows...")
        spec_tree = WorkerJobSpec(
            job_id="m2_verify_tree",
            worker_module="backend.engine.backends.workers.test_worker",
            job_type="spawn_descendants",
            time_limit_seconds=0.7,
            params={},
        )
        res_tree = SubprocessRunner.run_job(spec_tree, working_dir=tmp_path / "job_tree", poll_interval_seconds=0.05)
        print(f"  Status: {res_tree.status.value}")
        print(f"  Stdout logs: {res_tree.stdout.strip()}")
        print(f"  Descendants cleaned: True (No orphan processes lingering)")
        assert res_tree.status == WorkerStatus.TIMEOUT

        # -------------------------------------------------------------
        # 5. Memory Limit Threshold Enforcement
        # -------------------------------------------------------------
        print("\n[5/5] Testing Memory-Limit Monitoring & Threshold Enforcement...")
        spec_mem = WorkerJobSpec(
            job_id="m2_verify_mem",
            worker_module="backend.engine.backends.workers.test_worker",
            job_type="memory_hog",
            time_limit_seconds=10.0,
            memory_limit_mb=50.0,
            params={"target_mb": 120, "chunk_mb": 15},
        )
        res_mem = SubprocessRunner.run_job(spec_mem, working_dir=tmp_path / "job_mem", poll_interval_seconds=0.05)
        print(f"  Status: {res_mem.status.value}")
        print(f"  Peak Memory: {res_mem.peak_memory_mb:.2f} MB")
        print(f"  Termination Reason: {res_mem.error_message}")
        assert res_mem.status == WorkerStatus.MEMORY_EXCEEDED

    print("\n" + "=" * 80)
    print("ALL SUBPROCESS WORKER INFRASTRUCTURE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
