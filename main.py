"""
VINE-Agent v3: Main Entrypoint
Autonomous precision agriculture event loop, REST API, and CLI.

Usage:
    # 1. Start Autonomous Event Loop + API Server
    python main.py --serve --port 8000

    # 2. Interactive CLI Chatbot (Alert-Aware)
    python main.py --chat
    
    # 3. Benchmark queries
    python main.py --demo
"""

import os

# --- Apple Silicon Threading Fix ---
# Prevents FAISS/PyTorch from segfaulting or deadlocking when called inside
# daemon background threads (e.g. from the APScheduler enrichment loop)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import logging
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vine_agent")

DEMO_QUERIES = [
    "Should I irrigate Block A tomorrow?",
    "My leaves have yellow margins, is this related to any current alerts?",
]

def get_llm(provider: str = "groq", model: str = None, temperature: float = 0):
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            console.print("[red]GROQ_API_KEY missing.[/red]")
            sys.exit(1)
        model = model or "llama-3.3-70b-versatile"
        return ChatGroq(api_key=api_key, model=model, temperature=temperature)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "vllm":
        from langchain_core.language_models.llms import LLM
        import httpx
        class VLLMWrapper(LLM):
            server_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
            model_name = os.getenv("VLLM_MODEL", "/model")
            @property
            def _llm_type(self): return "vllm"
            def _call(self, prompt, stop=None):
                r = httpx.post(f"{self.server_url}/chat/completions", json={"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0}, timeout=120)
                return r.json()["choices"][0]["message"]["content"]
        return VLLMWrapper()
    raise ValueError(f"Unknown provider: {provider}")

def get_embed_model(device: str = "cpu"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": device})

def initialize_system(args):
    """Wires up the entire VINE v3 architecture."""
    console.print(Rule("[bold green]🍇 VINE-Agent v3 Autonomous Startup[/bold green]"))

    llm = get_llm(provider=args.llm, model=args.model)
    embed_model = get_embed_model(device=args.device)

    # Drone Imagery
    from drone_encoder import DroneImageryEncoder
    from sensor_stream import VINEYARD_BLOCKS
    console.print("[cyan]Initializing modalities...[/cyan]")
    drone_enc = DroneImageryEncoder()
    drone_blocks = drone_enc.generate_synthetic_ndvi_zones(
        blocks=[{"block": b["block"], "variety": b["variety"], "soil": b["soil"]} for b in VINEYARD_BLOCKS[:4]],
        n_flights=1
    )
    drone_text_blocks = drone_enc.encode_to_text_blocks(drone_blocks)
    
    # Knowledge Base
    from data_loader import build_vine_knowledge_base
    knowledge_texts, _ = build_vine_knowledge_base(drone_summaries=drone_text_blocks)
    
    # RAPTOR Index
    from raptor import Raptor
    if os.path.exists("raptor_index.pkl"):
        console.print("[cyan]Loading cached RAPTOR index (raptor_index.pkl)...[/cyan]")
        raptor_tree = Raptor.load("raptor_index.pkl", summariser_model=llm, embed_model=embed_model)
    else:
        console.print("[cyan]Building RAPTOR index... (takes ~7 LLM calls. If rate limited, wait and re-run)[/cyan]")
        raptor_tree = Raptor(docs=knowledge_texts, summariser_model=llm, embed_model=embed_model)
        raptor_tree.save("raptor_index.pkl")

    # Live Sensor Hook
    if getattr(args, "live_mqtt", False):
        from sensor_stream import MQTTSensorClient
        console.print("[cyan]Connecting to Live MQTT Broker at localhost:1883...[/cyan]")
        mqtt_client = MQTTSensorClient(broker="localhost", port=1883)
        mqtt_client.start()
        def sensor_fn(): return mqtt_client.get_latest_context()
    else:
        from sensor_stream import generate_live_sensor_context
        def sensor_fn(): return generate_live_sensor_context(n_blocks=4, scenario=args.sensor_scenario)

    # Autonomous Pipeline
    from proactive_agent import ProactivePipeline
    proactive_pipeline = ProactivePipeline(
        llm=llm, raptor_tree=raptor_tree, embed_model=embed_model,
        sensor_context_fn=sensor_fn, drone_blocks=drone_blocks
    )

    # Query Pipeline
    from query_agent import AlertAwareQueryAgent
    query_pipeline = AlertAwareQueryAgent(
        llm=llm, raptor_tree=raptor_tree, embed_model=embed_model,
        sensor_context_fn=sensor_fn, drone_blocks=drone_blocks
    )

    console.print("[bold green]✓ System Initialized[/bold green]")
    return proactive_pipeline, query_pipeline, sensor_fn


def run_server(args):
    """Starts the Event Scheduler and the FastAPI REST/WebSocket server."""
    proactive_pipeline, query_pipeline, sensor_fn = initialize_system(args)
    
    import api_server
    from scheduler import VINEScheduler
    
    # Inject dependencies into API
    api_server.query_pipeline = query_pipeline
    api_server.sensor_context_fn = sensor_fn

    # Start Event Scheduler
    scheduler = VINEScheduler(proactive_pipeline=proactive_pipeline)
    api_server.scheduler = scheduler
    scheduler.start()
    
    console.print(f"[bold green]Starting API Server on port {args.port}...[/bold green]")
    import uvicorn
    uvicorn.run(api_server.app, host="0.0.0.0", port=args.port)


def run_chat(args):
    _, query_pipeline, _ = initialize_system(args)
    console.print(Panel("[bold green]🍇 VINE-Agent v3 CLI[/bold green]\nAlert-Aware. Type exit to quit."))
    while True:
        try:
            q = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            if q.lower() in ("exit", "quit"): break
            res = query_pipeline.chat(q)
            
            cp = res.get("context_priority")
            if cp: console.print(f"[dim]Routing: {cp.primary} -> {cp.secondary}[/dim]")
            console.print(Markdown(res["answer"]))
        except KeyboardInterrupt: break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Start FastAPI + Autonomous Scheduler")
    parser.add_argument("--chat", action="store_true", help="CLI Chat")
    parser.add_argument("--demo", action="store_true", help="Run benchmark queries")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--llm", type=str, default="groq")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-hyde", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sensor-scenario", type=str, default="mixed")
    parser.add_argument("--live-mqtt", action="store_true", help="Connect to Live MQTT Broker on localhost")
    args = parser.parse_args()

    if args.serve:
        run_server(args)
    elif args.chat or args.demo:
        run_chat(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
