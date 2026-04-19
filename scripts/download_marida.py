from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
import sys

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not set. Add it to your .env file.")
    sys.exit(1)

snapshot_download(
    repo_id="HallowsYves/SeaPredictor",
    repo_type="dataset",
    local_dir="data/raw/MARIDA",
    token=HF_TOKEN,
)
