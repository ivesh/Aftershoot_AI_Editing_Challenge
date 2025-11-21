import os

class Config:
    # Base paths (update if needed)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "raw", "aftershoot_dataset", "Train")
    VAL_DIR = os.path.join(DATA_DIR, "raw", "aftershoot_dataset", "Validation")

def get_config():
    return Config()
