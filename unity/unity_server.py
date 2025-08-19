import json
import threading
import socket
import os

UNITY_HOST = "127.0.0.1"
UNITY_PORT = 5005

unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Absolute path of the current directory (unity folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODE_FILE = os.path.join(BASE_DIR, "current_mode.json")
MODE_LOCK = threading.Lock()

# Save the current control mode to a JSON file.
def set_current_mode(mode):
    with MODE_LOCK:
        try:
            with open(MODE_FILE, "w") as f:
                json.dump({"mode": mode}, f)
            print(f"⚙️ Control mode set to {mode}")
        except Exception as e:
            print(f"❌ Error writing mode file: {e}")

# Read the current control mode from the JSON file.
def get_current_mode():
    with MODE_LOCK:
        try:
            with open(MODE_FILE, "r") as f:
                data = json.load(f)
                return data.get("mode")
        except Exception:
            # Return None if file does not exist or is malformed
            return None

# Send a scalpel command to Unity via UDP.
def send_scalpel_command(state: str, source_id: int, extras: dict = None):
    current_mode = get_current_mode()
    if current_mode is not None and source_id != current_mode:
        print(f"🚫 Ignored command from source {source_id} (current mode: {current_mode})")
        return
    if source_id == 1 and state != "move":
        print("🚫 Manual control (source_id=1) can only send 'move' commands")
        return

    payload = {
        "scalpel_state": state,
        "id": source_id
    }
    if extras:
        payload.update(extras)

    try:
        json_msg = json.dumps(payload)
        unity_socket.sendto(json_msg.encode(), (UNITY_HOST, UNITY_PORT))
        print(f"🛰️ Sent to Unity: {json_msg}")
    except Exception as e:
        print(f"❌ Failed to send UDP message: {e}")

if __name__ == "__main__":
    print("Control mode input started. You can change the mode anytime.")
    while True:
        try:
            mode_input = input("Enter current control mode (1=manual, 2=HRV, 3=EEG): ").strip()
            mode = int(mode_input)
            if mode not in (1, 2, 3):
                print("Invalid input, please enter 1, 2, or 3")
                continue
            set_current_mode(mode)
        except Exception as e:
            print(f"Input error: {e}")
