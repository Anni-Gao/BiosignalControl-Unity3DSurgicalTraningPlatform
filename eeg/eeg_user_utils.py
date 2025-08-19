import os
import json

USER_DATA_DIR = "data/users"

def list_users():
    path = "data/users"
    if not os.path.exists(path):
        return []
    return [f.replace(".json", "") for f in os.listdir(path) if f.endswith(".json")]

def add_user(user_id):
    path = f"data/users/{user_id}.json"
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({"cognitiv_low": 0.0, "cognitiv_high": 1.0}, f)
        return True
    return False

def remove_user(user_id):
    path = f"data/users/{user_id}.json"
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
