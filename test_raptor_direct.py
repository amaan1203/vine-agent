import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
from main import get_embed_model
from raptor import Raptor
from proactive_agent import _EMBED_LOCK
import threading

logging.basicConfig(level=logging.INFO)

print("Loading embeddings...")
embed_model = get_embed_model("cpu")

print("Loading raptor...")
# Use the proper load method
raptor_tree = Raptor.load("raptor_index.pkl", summariser_model=None, embed_model=embed_model)

def run_test():
    print("[Thread] Testing direct retrieval...")
    with _EMBED_LOCK:
        nodes = raptor_tree.retrieve_collapsed("Show me drone data for Block A", top_k=2)
    print("[Thread] Nodes found:", len(nodes))
    print("[Thread] Retrieval SUCCESS")

t = threading.Thread(target=run_test)
t.start()
t.join()
print("Main thread finished")
