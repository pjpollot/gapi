from gapi import gapi

from fastapi.testclient import TestClient

client = TestClient(gapi)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_request():
    dat = {
        "train_x": [[1, 2, 3], [4, 5, 6]],
        "train_y": [1, 2],
        "x_to_predict": [[1, 1, 3], [5, 4, 3]], 
    }
    response = client.post("/predict", json=dat)
    assert response.status_code == 200