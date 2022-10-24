import os
import pickle
from fastapi import FastAPI


def load_models():
    model = pickle.load(open(os.path.join(os.getcwd(),
                                          'models',
                                          'model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(os.getcwd(),
                                           'models',
                                           'scaler.pkl'), 'rb'))
    return scaler, model


scaler, model = load_models()
app = FastAPI()


@app.get('/')
def home():
    return {"Hello": "World"}
