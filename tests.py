import requests

BASE_URL = "http://127.0.0.1:8000"

def test_predict():
    data = {"data": [[5.1, 3.5, 1.4, 0.2]]}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Predict response:", response.json())

def test_update_model(new_version=2):
    response = requests.post(f"{BASE_URL}/update-model?version={new_version}")
    if response.status_code == 200:
        print("Update response:", response.json())
        test_predict()
        response = requests.post(f"{BASE_URL}/update-model?version=1")
        print("Reverted to v1:", response.json() if response.status_code == 200 else response.text)
    else:
        print("Update failed:", response.text)
        return
    

if __name__ == "__main__":
    test_predict()
    test_update_model()
