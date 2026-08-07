from pydantic import BaseModel, EmailStr
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: int

class UserResponse(BaseModel):
    user_id: UUID
    full_name: str
    email: EmailStr
    role: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class DiagnosisRequest(BaseModel):
    input_text: str
    severity: int
    duration: int

class DiagnosisResponse(BaseModel):
    diagnosis_id: UUID
    input_text: str
    extracted_symptoms: Any
    predictions: Any
    validation_status: int
    prediction_timestamp: datetime

    class Config:
        from_attributes = True
