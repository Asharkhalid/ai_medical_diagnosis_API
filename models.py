import uuid
from sqlalchemy import Column, String, SmallInteger, DateTime, ForeignKey, Text, Boolean, Float
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
    is_active = Column(Boolean, nullable=False, default=True)
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

class Disease(Base):
    __tablename__ = "diseases"

    disease_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    disease_name = Column(String(150), unique=True, nullable=False, index=True)
    precautions = Column(Text, nullable=True)
    remedies = Column(Text, nullable=True)
    is_critical = Column(Boolean, nullable=False, default=False)

class SymptomDictionary(Base):
    __tablename__ = "symptom_dictionary"

    symptom_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    standard_name = Column(String(100), unique=True, nullable=False, index=True)
    synonyms = Column(JSONB, nullable=False, default=list)
    severity_weight = Column(SmallInteger, nullable=False, default=1)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(30), nullable=False)
    algorithm = Column(String(50), unique=True, nullable=False)
    trained_at = Column(DateTime, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision_score = Column(Float, nullable=False)
    recall_score = Column(Float, nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    event_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
