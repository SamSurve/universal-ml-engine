import sys
import json
import time
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Deterministic Test Worker for Subprocess Infrastructure Testing")
    parser.add_argument("--job-spec", required=True, help="Path to input job_spec.json")
    parser.add_argument("--result-path", required=True, help="Path to output result.json")

    args = parser.parse_args()

    spec_path = Path(args.job_spec)
    result_path = Path(args.result_path)

    if not spec_path.exists():
        print(f"Error: Job spec file {spec_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading job spec: {e}", file=sys.stderr)
        sys.exit(1)

    job_type = spec_data.get("job_type", "echo")
    params = spec_data.get("params", {})

    print(f"[TestWorker] Starting job_type={job_type}")

    # 1. Echo action
    if job_type == "echo":
        msg = params.get("message", "hello")
        print(f"STDOUT: {msg}")
        print(f"STDERR: notice_{msg}", file=sys.stderr)
        result_payload = {
            "status": "success",
            "payload": {"echo": msg, "computed_value": 42},
            "error_message": None,
        }
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        sys.exit(0)

    # 2. Sleep action (for timeout testing)
    elif job_type == "sleep":
        duration = float(params.get("duration", 5.0))
        print(f"[TestWorker] Sleeping for {duration} seconds...")
        time.sleep(duration)
        result_payload = {
            "status": "success",
            "payload": {"slept": duration},
            "error_message": None,
        }
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        sys.exit(0)

    # 3. Crash action (for crash testing)
    elif job_type == "crash":
        exit_code = int(params.get("exit_code", 42))
        err_msg = params.get("error_message", "Intentional test worker crash")
        print(f"FATAL: {err_msg}", file=sys.stderr)
        sys.exit(exit_code)

    # 4. Spawn descendants action (for process tree termination testing)
    elif job_type == "spawn_descendants":
        # Spawn 2 background child processes that sleep
        child1 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        child2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print(f"[TestWorker] Spawned child PIDs: {child1.pid}, {child2.pid}")
        # Parent sleeps so parent runner can terminate the whole tree
        time.sleep(120)
        sys.exit(0)

    # 5. Memory hog action (for memory limit testing)
    elif job_type == "memory_hog":
        target_mb = int(params.get("target_mb", 150))
        chunk_mb = int(params.get("chunk_mb", 15))
        allocations = []
        print(f"[TestWorker] Allocating up to {target_mb} MB in {chunk_mb} MB chunks...")
        allocated = 0
        while allocated < target_mb:
            allocations.append(bytearray(chunk_mb * 1024 * 1024))
            allocated += chunk_mb
            time.sleep(0.05)
        # Hold memory until runner terminates
        time.sleep(30)
        sys.exit(0)

    # 6. Invalid result action
    elif job_type == "invalid_result":
        print("[TestWorker] Writing invalid JSON to result file")
        result_path.write_text("{corrupt_json: invalid", encoding="utf-8")
        sys.exit(0)

    # 7. No result file action
    elif job_type == "no_result_file":
        print("[TestWorker] Exiting without writing result file")
        sys.exit(0)

    else:
        print(f"Unknown job_type: {job_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
