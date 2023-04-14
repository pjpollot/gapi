from fastapi.testclient import TestClient

from gapi import gapi

client = TestClient(gapi)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_request():
    train_x = [[1, 2, 3], [4, 5, 6]]
    train_y = [1, 2]
    x_to_predict = [[1, 1, 3], [5, 4, 3]]
    dat = {
        "train_x": train_x,
        "train_y": train_y,
        "x_to_predict": x_to_predict, 
    }
    response = client.post("/predict", json=dat)
    assert response.status_code == 200
    assert list(response.json().keys()) == ["x", "mean", "std"]
    assert len(response.json()["x"]) == len(response.json()["mean"]) == len(response.json()["std"]) == len(x_to_predict)