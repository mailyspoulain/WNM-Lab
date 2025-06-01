import os
import requests

def test_predict():
    # dossier où se situe test_request.py (la racine du projet)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    img_path = os.path.join(
        base_dir,             
        "test_structured",
        "Normal",
        "radio5.jpg"
    )

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    url = "http://127.0.0.1:5000/predict"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": f})

    print("Status Code:", resp.status_code)
    print("JSON response:", resp.json())

if __name__ == "__main__":
    test_predict()
