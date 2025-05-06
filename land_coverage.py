from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherData(BaseModel):
    temperature: Optional[float]
    humidity: Optional[float]
    precipitation: Optional[float]
    wind_speed: Optional[float]
    timestamp: Optional[datetime]

class SpectralData(BaseModel):
    index_name: str
    value: float
    unit: Optional[str] = None
    timestamp: Optional[datetime] = None

class LandCoverageResponse(BaseModel):
    question: str
    answer: str
    details: Dict[str, Any]
    confidence: Optional[float] = None
    sources: Optional[list] = None