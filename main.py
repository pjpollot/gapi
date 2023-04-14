import uvicorn

from gapi import gapi

if __name__ == "__main__":
    uvicorn.run(gapi, port=8000)