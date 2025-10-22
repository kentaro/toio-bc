#!/usr/bin/env python3
"""Start the WebSocket server."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from server.main import app


def main():
    """Run the WebSocket server."""
    print("[Server] Starting WebSocket server on http://0.0.0.0:8765")
    print("[Server] Open http://localhost:8765 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)


if __name__ == "__main__":
    main()
