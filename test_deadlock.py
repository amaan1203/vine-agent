import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
from prompt_toolkit import prompt
from main import get_embed_model
from raptor import Raptor
from proactive_agent import _EMBED_LOCK
import threading
import time

logging.basicConfig(level=logging.INFO)

embed_model = get_embed_model("cpu")
raptor_tree = Raptor.load("raptor_index.pkl", summariser_model=None, embed_model=embed_model)

def run_test():
    print("[Thread] Acquiring lock...")
    with _EMBED_LOCK:
        print("[Thread] Lock acquired. Retrieving...")
        nodes = raptor_tree.retrieve_collapsed("Show me drone data for Block A", top_k=2)
        print("[Thread] Retrieved.")

t = threading.Thread(target=run_test)
t.start()
t.join()
print("Success")
