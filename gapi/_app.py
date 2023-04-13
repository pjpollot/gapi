from fastapi import FastAPI
from gapi._objects import Input

gapi = FastAPI()

@gapi.get("/")
async def root():
    return {"message": "Welcome to gapi!"}

@gapi.post("/predict")
async def predict(input: Input):
    return {"prediction": "success"}