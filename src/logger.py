import logging
import os
import sys
import yaml


with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

log_dir = config["paths"]["log_dir"]
log_file = config["paths"]["log_file"]

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(module)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("OlistInferenceLogger")