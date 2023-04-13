from gapi import gapi
import uvicorn

if __name__ == "__main__":
    uvicorn.run(gapi, port=8000)