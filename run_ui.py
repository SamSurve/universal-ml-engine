import sys
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Universal ML Engine — Judge-Ready Interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 70)
    print("UNIVERSAL ML ENGINE — LOCAL WEB DASHBOARD")
    print(f"Server URL: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    uvicorn.run("backend.ui.server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
