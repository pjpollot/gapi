import torch

from fastapi import FastAPI

from gapi._objects import Input
from gapi._models import GPRegressionModel

gapi = FastAPI()

@gapi.get("/")
async def root():
    return {"message": "Welcome to gapi!"}

@gapi.post("/predict")
async def predict(input: Input):
    train_x = torch.tensor(input.train_x, dtype=torch.float)
    train_y = torch.tensor(input.train_y, dtype=torch.float)
    x_to_predict = torch.tensor(input.x_to_predict, dtype=torch.float)
    model = GPRegressionModel(train_x, train_y)
    posterior = model(x_to_predict)
    return {
        "x": x_to_predict.detach().tolist(),
        "mean": posterior.mean.detach().tolist(),
        "std": posterior.stddev.detach().tolist()
    }
