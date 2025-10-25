from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    symptoms: List[str]
    top_n: Optional[int] = 5

class DiseasePrediction(BaseModel):
    rank: int
    disease: str
    probability: float
    confidence: str
    description: str
    precautions: List[str]
    medications: List[str]
    diet: List[str]
    workout: List[str]

class PredictionResponse(BaseModel):
    input_symptoms: List[str]
    total_symptoms_matched: int
    predictions: List[DiseasePrediction]