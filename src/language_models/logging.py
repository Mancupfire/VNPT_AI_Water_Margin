import os
from datetime import datetime
import threading

file_lock = threading.Lock()

def log_to_file(filename, message, show_time = True, reset_file = False):
    log_dir = "logs/" + filename
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    if reset_file:
        with file_lock:
            with open(log_dir, 'w', encoding='utf-8') as f:
                f.write("")
            
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with file_lock:
        with open(log_dir, 'a', encoding='utf-8') as f:
            if show_time:
                f.write(f"{timestamp} - {message}\n")
            else:
                f.write(f"{message}\n")

def reset_file(filename):
    log_dir = "logs/" + filename
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    with open(log_dir, 'w', encoding='utf-8') as f:
        f.write("")