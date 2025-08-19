import json
import os
import numpy as np

def send_json(ws, method, params=None, id=None):
    msg = {"jsonrpc": "2.0", "method": method}
    if id: msg["id"] = id
    if params: msg["params"] = params
    ws.send(json.dumps(msg))

def calculate_ranges_with_std(values, k=1.0):
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    lower = max(0, mean - k * std)
    upper = mean + k * std
    return round(lower, 2), round(upper, 2)

def save_user_ranges(range_file, ranges):
    with open(range_file, "w") as f:
        json.dump(ranges, f)

def load_user_ranges(range_file):
    if not os.path.exists(range_file):
        return None
    with open(range_file, "r") as f:
        return json.load(f)