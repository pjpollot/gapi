# GAPI

A simple REST API for Gaussian Process models written in `Python 3.9` using [FastAPI](https://fastapi.tiangolo.com/).

## Installation

On the root of the repository, run:

```bash
pip install .
```

For developers, poetry is used to manage the dependencies. To install the dependencies, run:

```bash
poetry install 
```

## Usage

To run the API, use the following command:

```bash
python main.py
```

API call through HTTP request:

```
POST {your url}/predict

with body: {
"train_x": <the inputs where the GP is trained, type: list[list[float]]>,
"train_y": <the labels which the GP is trained, type: list[float]>,
"x_to_predict": <where to predict, type: list[list[float]]>
}

and returns (if successful): {
"x": <the input 'x_to_predict'>,
"mean": <posterior mean of type list[float]>
"std": <the standard deviation of type list[float]>
}
```

## Unit testing 

To run the unit tests, use the following command on the root of this repository:

```bash
pytest
```
