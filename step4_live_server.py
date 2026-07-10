"""
STEP 4 - Live streaming server
=============================
A minimal FastAPI + WebSocket server that exposes live pose data for the avatar
front-end.
"""

import asyncio
import json
from pathlib import Path
from typing import Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Arm Reachability Tracker")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

clients: Set[WebSocket] = set()


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(Path(__file__).parent / "static" / "avatar.html")


@app.get("/health")
def health(request: Request):
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        return HTMLResponse(
            """
            <html>
              <head>
                <title>Arm Reachability Tracker</title>
                <style>
                  body { font-family: Arial, sans-serif; margin: 2rem; background: #111; color: #f5f5f5; }
                  .card { background: #1f1f1f; padding: 1.5rem; border-radius: 12px; display: inline-block; }
                  code { background: #2b2b2b; padding: 0.2rem 0.4rem; border-radius: 4px; }
                </style>
              </head>
              <body>
                <div class="card">
                  <h1>Arm Reachability Tracker</h1>
                  <p>Server is running correctly.</p>
                  <p>WebSocket endpoint: <code>/ws</code></p>
                  <p>Open the avatar page at <code>/</code>.</p>
                </div>
              </body>
            </html>
            """
        )
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse("", status_code=204)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await broadcast_to_clients(message, sender=websocket)
    except WebSocketDisconnect:
        clients.discard(websocket)


async def broadcast_to_clients(message: str, sender: WebSocket):
    if not clients:
        return

    dead_clients = set()
    for client in list(clients):
        if client is sender:
            continue
        try:
            await client.send_text(message)
        except Exception:
            dead_clients.add(client)
    clients.difference_update(dead_clients)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
