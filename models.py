import uuid
from sqlalchemy import Column, String, SmallInteger, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(SmallInteger, nullable=False) # 1=Patient, 2=Doctor, 3=Admin
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    diagnoses = relationship("DiagnosisHistory", back_populates="patient")

class DiagnosisHistory(Base):
    __tablename__ = "diagnosis_history"

    diagnosis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    input_text = Column(Text, nullable=False)
    extracted_symptoms = Column(JSONB, nullable=False)
    severity = Column(SmallInteger, nullable=False)
    duration = Column(SmallInteger, nullable=False)
    predictions = Column(JSONB, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(30), nullable=False)
    prediction_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    validation_status = Column(SmallInteger, nullable=False) # 0=Pending, 1=Validated, 2=Rejected

    patient = relationship("User", back_populates="diagnoses")
    feedbacks = relationship("DiagnosisFeedback", back_populates="diagnosis")

class DiagnosisFeedback(Base):
    __tablename__ = "diagnosis_feedback"

    feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    diagnosis_id = Column(UUID(as_uuid=True), ForeignKey("diagnosis_history.diagnosis_id"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    comments = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    diagnosis = relationship("DiagnosisHistory", back_populates="feedbacks")

class ExpertRule(Base):
    __tablename__ = "expert_rules"

    rule_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    condition = Column(String(255), nullable=False)
    target = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    priority = Column(SmallInteger, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    doctor = relationship("User")
