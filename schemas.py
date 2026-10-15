from pydantic import BaseModel, EmailStr
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class AdminUserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: int

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AdminUserEdit(BaseModel):
    full_name: str
    email: EmailStr

class UserStatusRequest(BaseModel):
    is_active: bool

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

class AuditLogResponse(BaseModel):
    log_id: UUID
    user_id: Optional[UUID]
    event_type: str
    description: str
    created_at: datetime

    class Config:
        from_attributes = True

class SynonymRequest(BaseModel):
    phrase: str
    target: str

class DiagnosisResponse(BaseModel):
    diagnosis_id: UUID
    input_text: str
    extracted_symptoms: Any
    predictions: Any
    validation_status: int
    prediction_timestamp: datetime

    class Config:
        from_attributes = True

class ValidationRequest(BaseModel):
    status: int

class FeedbackRequest(BaseModel):
    comments: str

class ExpertRuleRequest(BaseModel):
    condition: str
    target: str
    action: str
    priority: int

class ExpertRuleResponse(BaseModel):
    rule_id: UUID
    condition: str
    target: str
    action: str
    priority: int
    created_by: UUID
    created_at: datetime

    class Config:
        from_attributes = True
