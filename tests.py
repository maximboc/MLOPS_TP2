import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_predict(n=5):
    data = {"data": [[5.1, 3.5, 1.4, 0.2]]}
    for i in range(n):
        response = requests.post(f"{BASE_URL}/predict", json=data)
        if response.status_code == 200:
            print(f"[{i+1}] Predict response:", response.json())
        else:
            print(f"[{i+1}] Predict failed:", response.text)
        time.sleep(0.5)


def test_update_model(new_version=2):
    response = requests.post(f"{BASE_URL}/update-model?version={new_version}")
    if response.status_code == 200:
        print("✅ Update response:", response.json())
    else:
        print("❌ Update failed:", response.text)


def test_accept_next_model():
    response = requests.post(f"{BASE_URL}/accept-next-model")
    if response.status_code == 200:
        print("✅ Accept response:", response.json())
    else:
        print("❌ Accept failed:", response.text)


if __name__ == "__main__":
    print("\n--- Initial prediction with current model ---")
    test_predict()

    print("\n--- Updating next model to version 2 ---")
    test_update_model(new_version=2)

    print("\n--- Predictions (mix of current and next) ---")
    test_predict()

    print("\n--- Accepting next model as current ---")
    test_accept_next_model()

    print("\n--- Predictions after accept (both models same again) ---")
    test_predict()

    print("\n--- Reverting next model back to v1 for cleanup ---")
    test_update_model(new_version=1)
