#  VINE-Agent: Agentic RAG for Precision Agriculture

> **GSoC 2026 | VINE / AI/ML Models for Agricultural Analytics on NRP**


VINE-Agent is an agentic RAG system that combines **RAPTOR**, **ColBERT**, **HyDE**, and **ReWOO** to provide intelligent decision-support for precision viticulture at Iron Horse Vineyards.


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

# Interactive chatbot mode
python main.py --chat

# live-mqtt mode 
python main.py --live-mqtt

# Load custom docs
python main.py --docs path/to/agro_docs/ --query "When to irrigate Pinot Noir?"
```

## Benchmark Queries I have tested on

1. "Should I irrigate Block C (Pinot Noir) tomorrow morning given forecast temperatures?"
2. "My vines show pale leaves and drooping — what could cause this?"
3. "What NDVI threshold indicates Botrytis risk in Chardonnay?"
4. "Compare water stress indicators for véraison vs post-harvest periods."
5. "What is the optimal deficit irrigation strategy for Cabernet Sauvignon during berry set?"
