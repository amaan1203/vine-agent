# 🍇 VINE-Agent: Agentic RAG for Precision Agriculture

> **GSoC 2026 | VINE / AI/ML Models for Agricultural Analytics on NRP**
> Inspired by [ReCOR-RAG](https://github.com/recor-rag) codebase architecture

VINE-Agent is a modular agentic RAG system that combines **RAPTOR**, **ColBERT**, **HyDE**, and **ReWOO** to provide intelligent decision-support for precision viticulture at Iron Horse Vineyards.

## Architecture

```
Query (farmer)
    │
    ▼ HyDE expander
Hypothetical agronomic answer → embedding
    │
    ▼ RAPTOR index (UMAP+GMM clustering)
ColBERT re-ranked top-k docs
    │
    ▼ ReWOO planner (LangGraph)
Plan (Planner) → parallel tool calls (Worker) → synthesis (Solver)
    │
    ▼
Cited agricultural recommendation
```

## Pipeline Graph

```
START → query_rewrite → make_raptor → planner → worker → solver → summary → END
```

## Data Sources (Demo Mode)
- **NDVI Sensor Transcripts** – synthetic agricultural knowledge base from crop health bulletins
- **UC Davis Viticulture PDFs** – open extension publications on grapevine physiology
- **Weather & Irrigation Records** – publicly available CIMIS/AZMET station data

## Quick Start

```bash
pip install -r requirements.txt

# Set your API key (uses Groq by default — fast & free tier available)
export GROQ_API_KEY=your_key_here

# Demo with built-in agricultural knowledge base
python main.py --demo

# Interactive chatbot mode
python main.py --chat

# Load custom docs
python main.py --docs path/to/agro_docs/ --query "When to irrigate Pinot Noir?"
```

## Project Structure

```
vine-agent/
├── main.py              # CLI entrypoint
├── agent.py             # ReWOO LangGraph pipeline
├── raptor.py            # RAPTOR hierarchical indexer (ported from ReCOR-RAG)
├── hyde.py              # HyDE query expander
├── retriever.py         # ColBERT + FAISS hybrid retriever
├── data_loader.py       # Agricultural dataset loaders
├── prompts.py           # Agricultural-domain prompts (Planner, Solver, Summary)
├── requirements.txt
└── data/
    └── agro_knowledge/  # Sample agricultural knowledge base
```

## Benchmark Queries (from proposal)

1. "Should I irrigate Block C (Pinot Noir) tomorrow morning given forecast temperatures?"
2. "My vines show pale leaves and drooping — what could cause this?"
3. "What NDVI threshold indicates Botrytis risk in Chardonnay?"
4. "Compare water stress indicators for véraison vs post-harvest periods."
5. "What is the optimal deficit irrigation strategy for Cabernet Sauvignon during berry set?"
