from gapi import gapi
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:gapi", reload=True, port=8000)