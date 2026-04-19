from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# input schema
class Input(BaseModel):
    lat: float
    lon: float

@app.post("/predict")
def predict(data: Input):
    # 🔥 replace this with your model later
    return {
        "lat": data.lat,
        "lon": data.lon,
        "confidence": 0.87
    }