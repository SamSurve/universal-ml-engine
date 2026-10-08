import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import psutil

from backend.engine.contracts.schemas import (
    WorkerJobSpec,
    WorkerResult,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


def terminate_process_tree(pid: int, timeout: float = 3.0) -> None:
    """
    Terminates a process and all of its descendant processes and threads cleanly on Windows/Linux.
    Ensures no orphan worker or OpenMP background threads linger in memory.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    # On Windows, try taskkill first for robust native tree termination
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass

    # Use psutil to guarantee all children are signaled
    for child in children:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    # Wait for processes to disappear
    _, still_alive = psutil.wait_procs(children + [parent], timeout=timeout)
    for p in still_alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def measure_process_tree_memory_mb(pid: int) -> float:
    """
    Computes total RSS memory (in Megabytes) consumed by the process and all its descendants.
    """
    try:
        parent = psutil.Process(pid)
        total_rss = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss / (1024.0 * 1024.0)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


class SubprocessRunner:
    """
    Parent-side execution manager that runs isolated worker jobs in separate processes
    with strict timeout enforcement, memory-limit monitoring, and robust process tree cleanup.
    """

    @classmethod
    def run_job(
        cls,
        job_spec: WorkerJobSpec,
        working_dir: Optional[Path] = None,
        poll_interval_seconds: float = 0.05,
        python_executable: Optional[Union[str, Path]] = None,
    ) -> WorkerResult:
        """
        Executes a worker job specification in an isolated subprocess.

        Parameters:
            job_spec: Job specification defining module, timeout, memory limits, and params.
            working_dir: Working directory for temporary IPC files. Defaults to system temp.
            poll_interval_seconds: Polling sleep interval during execution.
            python_executable: Custom Python interpreter path to use. Defaults to job_spec.python_executable or sys.executable.

        Returns:
            Structured WorkerResult.
        """
        run_dir = working_dir or Path(os.environ.get("TEMP", ".")) / f"worker_{job_spec.job_id}_{int(time.time()*1000)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        spec_file = run_dir / "job_spec.json"
        result_file = run_dir / "job_result.json"

        # Determine Python executable
        raw_python = python_executable or getattr(job_spec, "python_executable", None) or sys.executable
        exec_python = str(Path(raw_python).resolve())

        # Serialize job spec to JSON
        spec_dict = {
            "job_id": job_spec.job_id,
            "worker_module": job_spec.worker_module,
            "job_type": job_spec.job_type,
            "time_limit_seconds": job_spec.time_limit_seconds,
            "memory_limit_mb": job_spec.memory_limit_mb,
            "python_executable": getattr(job_spec, "python_executable", None),
            "params": job_spec.params,
        }
        spec_file.write_text(json.dumps(spec_dict, indent=2), encoding="utf-8")

        # Build execution arguments (Strictly argument list, never shell=True)
        if job_spec.worker_module.endswith(".py"):
            target_path = str(Path(job_spec.worker_module).resolve())
            cmd_args = [
                exec_python,
                target_path,
                "--job-spec",
                str(spec_file.resolve()),
                "--result-path",
                str(result_file.resolve()),
            ]
        else:
            cmd_args = [
                exec_python,
                "-m",
                job_spec.worker_module,
                "--job-spec",
                str(spec_file.resolve()),
                "--result-path",
                str(result_file.resolve()),
            ]

        # Add project root to PYTHONPATH for the subprocess
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        current_ppath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_ppath}" if current_ppath else project_root

        start_time = time.time()
        peak_memory_mb = 0.0

        stdout_log_path = run_dir / "worker_stdout.log"
        stderr_log_path = run_dir / "worker_stderr.log"

        try:
            with open(stdout_log_path, "w", encoding="utf-8", errors="replace") as out_f, \
                 open(stderr_log_path, "w", encoding="utf-8", errors="replace") as err_f:
                proc = subprocess.Popen(
                    cmd_args,
                    stdin=subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    cwd=str(run_dir),
                    env=env,
                )

                worker_pid = proc.pid
                terminated_status: Optional[WorkerStatus] = None
                term_error: Optional[str] = None

                # Active monitoring loop
                while proc.poll() is None:
                    elapsed = time.time() - start_time
                    current_mem = measure_process_tree_memory_mb(worker_pid)
                    peak_memory_mb = max(peak_memory_mb, current_mem)

                    # 1. Memory limit enforcement
                    if job_spec.memory_limit_mb is not None and current_mem > job_spec.memory_limit_mb:
                        logger.warning(
                            f"Worker PID {worker_pid} exceeded memory limit: {current_mem:.1f} MB > {job_spec.memory_limit_mb:.1f} MB. Terminating."
                        )
                        terminate_process_tree(worker_pid)
                        terminated_status = WorkerStatus.MEMORY_EXCEEDED
                        term_error = (
                            f"Process tree exceeded memory limit ({current_mem:.1f} MB > {job_spec.memory_limit_mb:.1f} MB)"
                        )
                        break

                    # 2. Timeout enforcement
                    if elapsed > job_spec.time_limit_seconds:
                        logger.warning(
                            f"Worker PID {worker_pid} exceeded time limit: {elapsed:.2f}s > {job_spec.time_limit_seconds:.2f}s. Terminating."
                        )
                        terminate_process_tree(worker_pid)
                        terminated_status = WorkerStatus.TIMEOUT
                        term_error = f"Process timed out after {job_spec.time_limit_seconds:.2f}s (elapsed: {elapsed:.2f}s)"
                        break

                    time.sleep(poll_interval_seconds)

                proc.wait()
        except Exception as e:
            logger.error(f"Failed to launch worker subprocess: {e}")
            return WorkerResult(
                job_id=job_spec.job_id,
                status=WorkerStatus.FAILED,
                exit_code=None,
                runtime_seconds=0.0,
                peak_memory_mb=0.0,
                error_message=f"Subprocess spawn failed: {e}",
            )

        # Capture outputs from log files
        stdout = stdout_log_path.read_text(encoding="utf-8", errors="replace") if stdout_log_path.exists() else ""
        stderr = stderr_log_path.read_text(encoding="utf-8", errors="replace") if stderr_log_path.exists() else ""
        runtime = round(time.time() - start_time, 4)

        # Handle forced termination states
        if terminated_status is not None:
            return WorkerResult(
                job_id=job_spec.job_id,
                status=terminated_status,
                exit_code=proc.returncode,
                runtime_seconds=runtime,
                peak_memory_mb=round(peak_memory_mb, 2),
                stdout=stdout,
                stderr=stderr,
                error_message=term_error,
            )

        # Handle non-zero exit code (Crash)
        if proc.returncode != 0:
            err_msg = stderr.strip() if stderr else f"Worker crashed with exit code {proc.returncode}"
            return WorkerResult(
                job_id=job_spec.job_id,
                status=WorkerStatus.CRASHED,
                exit_code=proc.returncode,
                runtime_seconds=runtime,
                peak_memory_mb=round(peak_memory_mb, 2),
                stdout=stdout or "",
                stderr=stderr or "",
                error_message=err_msg,
            )

        # Handle normal completion -> Parse result.json
        if not result_file.exists():
            return WorkerResult(
                job_id=job_spec.job_id,
                status=WorkerStatus.FAILED,
                exit_code=0,
                runtime_seconds=runtime,
                peak_memory_mb=round(peak_memory_mb, 2),
                stdout=stdout or "",
                stderr=stderr or "",
                error_message="Worker completed with exit code 0 but result.json was not generated.",
            )

        try:
            result_data = json.loads(result_file.read_text(encoding="utf-8"))
            status_str = result_data.get("status", "success")
            status = WorkerStatus.SUCCESS if status_str == "success" else WorkerStatus.FAILED
            payload = result_data.get("payload", {})
            err_msg = result_data.get("error_message")

            return WorkerResult(
                job_id=job_spec.job_id,
                status=status,
                exit_code=0,
                runtime_seconds=runtime,
                peak_memory_mb=round(peak_memory_mb, 2),
                stdout=stdout or "",
                stderr=stderr or "",
                payload=payload,
                error_message=err_msg,
            )
        except Exception as e:
            return WorkerResult(
                job_id=job_spec.job_id,
                status=WorkerStatus.FAILED,
                exit_code=0,
                runtime_seconds=runtime,
                peak_memory_mb=round(peak_memory_mb, 2),
                stdout=stdout or "",
                stderr=stderr or "",
                error_message=f"Failed to parse worker result.json: {e}",
            )
