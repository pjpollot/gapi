import torch
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from gapi._objects import Input
from gapi._models import GPRegressionModel

gapi = FastAPI()

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

@gapi.get("/", response_class=HTMLResponse)
def root():
    return templates.TemplateResponse("index.html", {"request": {}})

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
