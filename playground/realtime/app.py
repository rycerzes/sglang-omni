#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Lightweight static server for the realtime voice chat playground.

Serves the frontend files and proxies the API base URL into the HTML
so the browser knows where to open the WebSocket.
"""

import argparse
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent

app = FastAPI(title="sglang-omni-realtime-playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index() -> HTMLResponse:
    API_BASE = os.environ.get("SGLANG_OMNI_API_BASE", "")
    html = (FRONTEND_DIR / "index.html").read_text()
    if API_BASE:
        injection = f'<script>window.SGLANG_OMNI_API_BASE = "{API_BASE}";</script>'
        html = html.replace("<head>", f"<head>{injection}", 1)
    return HTMLResponse(html)


# Mount static files after the explicit route so / is handled by index().
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Voice Chat Playground")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
