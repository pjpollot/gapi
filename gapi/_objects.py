from pydantic import BaseModel

class Input(BaseModel):
    train_x: list[list[float]]
    train_y: list[float]
    x_to_predict: list[list[float]] 