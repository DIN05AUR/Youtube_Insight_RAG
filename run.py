#!/usr/bin/env python3
# run.py — starts the FastAPI server

import uvicorn

if __name__ == "__main__":
    PORT = 8000

    print("\n" + "=" * 45)
    print("  Youtube Insight RAG — Starting server...")
    print("=" * 45)
    print(f"  Local:    http://localhost:{PORT}")
    print(f"  Network:  http://0.0.0.0:{PORT}")
    print(f"  API docs: http://localhost:{PORT}/docs")
    print("=" * 45)
    print("  Press CTRL+C to stop\n")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,   # set True during development if you're editing api.py
        log_level="warning",
    )