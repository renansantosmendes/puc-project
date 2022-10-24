import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class Data(BaseModel):
    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float


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


@app.post('/api/predict')
def predict(data: Data):
    received_data = np.array([
        data.baseline_value,
        data.accelerations,
        data.fetal_movement,
        data.uterine_contractions,
        data.light_decelerations,
        data.severe_decelerations,
        data.prolongued_decelerations
    ]).reshape(1, -1)
    return {'y_pred': model.predict(scaler.transform(received_data))[0]}