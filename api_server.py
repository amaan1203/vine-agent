import logging
import asyncio
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from alert_store import AlertStore

logger = logging.getLogger(__name__)

alert_store = AlertStore()
query_pipeline = None         # Set by main.py
sensor_context_fn = None      # Set by main.py
scheduler = None              # Set by main.py


app = FastAPI(
    title="VINE-Agent v3 API", 
    description="Autonomous Precision Agriculture System",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/api/v1/alerts")
def get_alerts(limit: int = 50):
    try:
        data = alert_store.get_active_alerts(limit=limit)
        return {"count": len(data), "alerts": data}
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail="Database error.")


@app.get("/api/v1/alerts/enriched")
def get_enriched_alerts(limit: int = 20):
    try:
        import sqlite3
        with sqlite3.connect(alert_store.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM alerts WHERE recommendation IS NOT NULL ORDER BY triggered_at DESC LIMIT ?",
                (limit,)
            )
            data = [dict(r) for r in cur.fetchall()]
        return {"count": len(data), "enriched_alerts": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/status")
def get_system_status():
    try:
        import sqlite3
        with sqlite3.connect(alert_store.sqlite_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            enriched = conn.execute("SELECT COUNT(*) FROM alerts WHERE recommendation IS NOT NULL").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM alerts WHERE recommendation IS NULL AND status='TRIGGERED'").fetchone()[0]
            critical = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity='CRITICAL'").fetchone()[0]
            high = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity='HIGH'").fetchone()[0]
            latest = conn.execute("SELECT triggered_at FROM alerts ORDER BY triggered_at DESC LIMIT 1").fetchone()

        return {
            "system": "VINE-Agent v3 Autonomous",
            "scheduler": "ACTIVE",
            "total_alerts": total,
            "enriched_alerts": enriched,
            "pending_enrichment": pending,
            "by_severity": {"CRITICAL": critical, "HIGH": high},
            "enrichment_rate_pct": round(enriched / total * 100, 1) if total else 0,
            "last_alert_at": latest[0] if latest else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sensor/{block}")
def get_sensor(block: str):
    # In production, this pulls from Redis
    if sensor_context_fn:
        return {"block": block, "sensor_context": sensor_context_fn()}
    raise HTTPException(status_code=404, detail="Live sensor feed unconnected.")


@app.post("/api/v1/query")
def submit_query(req: QueryRequest):
    if not query_pipeline:
        raise HTTPException(status_code=503, detail="Query agent not initialized.")
    
    logger.info(f"API received query: {req.query}")
    try:
        response = query_pipeline.chat(req.query)
        return {"query": req.query, "response": response}
    except Exception as e:
        logger.error(f"Query failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
