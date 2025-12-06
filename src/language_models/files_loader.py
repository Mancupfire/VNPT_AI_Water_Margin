import numpy as np
import pandas as pd
import pickle
import os
    
def load_csv(path):
    return pd.read_csv(path, encoding='utf-8', low_memory=False)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, encoding='utf-8', index=False)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)

def save_txt(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(lines)