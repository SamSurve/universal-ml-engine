import re
import pytest
import psutil
from pathlib import Path

from backend.engine.contracts.schemas import WorkerJobSpec, WorkerStatus, WorkerResult
from backend.engine.backends.subprocess_runner import SubprocessRunner, terminate_process_tree


def test_worker_successful_execution(tmp_path):
    """Verify that a standard worker job executes successfully and returns expected payload."""
    spec = WorkerJobSpec(
        job_id="test_success_001",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="echo",
        time_limit_seconds=10.0,
        params={"message": "hello_from_parent"},
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path)

    assert isinstance(result, WorkerResult)
    assert result.status == WorkerStatus.SUCCESS
    assert result.exit_code == 0
    assert result.payload.get("echo") == "hello_from_parent"
    assert result.payload.get("computed_value") == 42
    assert result.runtime_seconds > 0.0
    assert result.peak_memory_mb > 0.0


def test_worker_stdout_stderr_capture(tmp_path):
    """Verify that stdout and stderr streams are faithfully captured in WorkerResult."""
    spec = WorkerJobSpec(
        job_id="test_stream_002",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="echo",
        time_limit_seconds=10.0,
        params={"message": "stream_test_123"},
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path)

    assert result.status == WorkerStatus.SUCCESS
    assert "STDOUT: stream_test_123" in result.stdout
    assert "STDERR: notice_stream_test_123" in result.stderr


def test_worker_crash_handling(tmp_path):
    """Verify that a worker crash (non-zero exit) is caught with CRASHED status and stderr captured."""
    spec = WorkerJobSpec(
        job_id="test_crash_003",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="crash",
        time_limit_seconds=10.0,
        params={"exit_code": 42, "error_message": "Segmentation fault simulation"},
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path)

    assert result.status == WorkerStatus.CRASHED
    assert result.exit_code == 42
    assert "Segmentation fault simulation" in result.stderr
    assert "Segmentation fault simulation" in (result.error_message or "")


def test_worker_timeout_and_forced_termination(tmp_path):
    """Verify that a hung worker is terminated after timeout and marked with TIMEOUT status."""
    spec = WorkerJobSpec(
        job_id="test_timeout_004",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="sleep",
        time_limit_seconds=0.6,
        params={"duration": 15.0},
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path, poll_interval_seconds=0.05)

    assert result.status == WorkerStatus.TIMEOUT
    assert result.runtime_seconds < 4.0
    assert "timed out" in (result.error_message or "").lower()


def test_worker_descendant_process_termination(tmp_path):
    """Verify that worker and all of its spawned child processes are killed on timeout."""
    spec = WorkerJobSpec(
        job_id="test_descendants_005",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="spawn_descendants",
        time_limit_seconds=0.8,
        params={},
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path, poll_interval_seconds=0.05)

    assert result.status == WorkerStatus.TIMEOUT

    # Extract child PIDs from worker stdout: "[TestWorker] Spawned child PIDs: 1234, 5678"
    pids = re.findall(r"Spawned child PIDs:\s*(\d+),\s*(\d+)", result.stdout)
    if pids:
        child1_pid = int(pids[0][0])
        child2_pid = int(pids[0][1])

        # Extra safety check: ensure neither child process is still alive
        assert not psutil.pid_exists(child1_pid), f"Orphan descendant child PID {child1_pid} is still alive!"
        assert not psutil.pid_exists(child2_pid), f"Orphan descendant child PID {child2_pid} is still alive!"


def test_worker_memory_limit_termination(tmp_path):
    """Verify that worker is terminated when exceeding memory limit and marked MEMORY_EXCEEDED."""
    spec = WorkerJobSpec(
        job_id="test_mem_006",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="memory_hog",
        time_limit_seconds=10.0,
        memory_limit_mb=45.0,  # 45 MB threshold
        params={"target_mb": 120, "chunk_mb": 15},
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path, poll_interval_seconds=0.05)

    assert result.status == WorkerStatus.MEMORY_EXCEEDED
    assert result.peak_memory_mb >= 45.0
    assert "exceeded memory limit" in (result.error_message or "").lower()


def test_worker_invalid_result_handling(tmp_path):
    """Verify that corrupted JSON result files are caught and marked with FAILED status."""
    spec = WorkerJobSpec(
        job_id="test_corrupt_007",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="invalid_result",
        time_limit_seconds=5.0,
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path)

    assert result.status == WorkerStatus.FAILED
    assert "failed to parse" in (result.error_message or "").lower()


def test_worker_missing_result_file_handling(tmp_path):
    """Verify that workers that exit 0 without writing a result file are marked FAILED."""
    spec = WorkerJobSpec(
        job_id="test_missing_008",
        worker_module="backend.engine.backends.workers.test_worker",
        job_type="no_result_file",
        time_limit_seconds=5.0,
    )

    result = SubprocessRunner.run_job(spec, working_dir=tmp_path)

    assert result.status == WorkerStatus.FAILED
    assert "result.json was not generated" in (result.error_message or "")
