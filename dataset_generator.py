import csv
import random

# Target diseases
diseases = [
    "Common Cold", "Influenza", "COVID-19", "Acute Migraine",
    "Gastroenteritis", "Dengue Fever", "Typhoid", "Pneumonia",
    "Bronchitis", "Tuberculosis", "Malaria", "Asthma",
    "Anemia", "Diabetes Type 2", "Hypertension",
    "Chickenpox", "Hepatitis B", "Urinary Tract Infection",
    "Rheumatoid Arthritis", "GERD", "Sinusitis",
    "Kidney Stones", "Appendicitis", "Measles", "Skin Allergy"
]

# All possible symptoms across the diseases
symptoms = [
    "fever", "headache", "cough", "fatigue", "abdominal_pain",
    "nausea", "vomiting", "sore_throat", "shortness_of_breath",
    "body_pain", "chills", "loss_of_taste", "loss_of_smell",
    "diarrhea", "chest_pain", "wheezing", "dizziness",
    "weight_loss", "sweating", "pale_skin", "frequent_urination",
    "rash", "jaundice", "joint_pain", "runny_nose", "back_pain",
    "itching", "blurred_vision", "swollen_lymph_nodes",
    "loss_of_appetite", "dark_urine", "burning_urination",
    "bloating", "acid_reflux", "muscle_cramps", "palpitations"
]

# Disease symptom profiles (which symptoms they typically present)
disease_profiles = {
    "Common Cold": ["cough", "sore_throat", "fatigue", "body_pain", "runny_nose"],
    "Influenza": ["fever", "chills", "body_pain", "fatigue", "headache", "cough"],
    "COVID-19": ["fever", "cough", "loss_of_taste", "loss_of_smell", "fatigue", "shortness_of_breath"],
    "Acute Migraine": ["headache", "nausea", "dizziness", "vomiting", "blurred_vision"],
    "Gastroenteritis": ["abdominal_pain", "nausea", "vomiting", "diarrhea", "fever", "bloating"],
    "Dengue Fever": ["fever", "headache", "body_pain", "fatigue", "nausea", "rash", "joint_pain"],
    "Typhoid": ["fever", "headache", "abdominal_pain", "fatigue", "weight_loss", "loss_of_appetite"],
    "Pneumonia": ["cough", "fever", "chills", "shortness_of_breath", "chest_pain"],
    "Bronchitis": ["cough", "fatigue", "shortness_of_breath", "chest_pain", "wheezing"],
    "Tuberculosis": ["cough", "weight_loss", "fatigue", "fever", "sweating", "loss_of_appetite"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea", "body_pain", "muscle_cramps"],
    "Asthma": ["shortness_of_breath", "chest_pain", "wheezing", "cough"],
    "Anemia": ["fatigue", "pale_skin", "dizziness", "chest_pain", "shortness_of_breath", "palpitations"],
    "Diabetes Type 2": ["frequent_urination", "fatigue", "weight_loss", "blurred_vision"],
    "Hypertension": ["headache", "dizziness", "fatigue", "chest_pain", "palpitations"],
    "Chickenpox": ["rash", "itching", "fever", "fatigue", "headache", "loss_of_appetite"],
    "Hepatitis B": ["jaundice", "dark_urine", "fatigue", "loss_of_appetite", "nausea", "abdominal_pain"],
    "Urinary Tract Infection": ["burning_urination", "frequent_urination", "back_pain", "nausea", "fever"],
    "Rheumatoid Arthritis": ["joint_pain", "fatigue", "fever", "swollen_lymph_nodes", "muscle_cramps"],
    "GERD": ["acid_reflux", "bloating", "nausea", "chest_pain", "sore_throat"],
    "Sinusitis": ["runny_nose", "headache", "fever", "fatigue", "sore_throat", "loss_of_appetite"],
    "Kidney Stones": ["back_pain", "nausea", "vomiting", "frequent_urination", "burning_urination"],
    "Appendicitis": ["abdominal_pain", "fever", "nausea", "vomiting", "loss_of_appetite"],
    "Measles": ["rash", "fever", "runny_nose", "cough", "swollen_lymph_nodes"],
    "Skin Allergy": ["rash", "itching", "swollen_lymph_nodes", "runny_nose", "fatigue"],
}

filename = 'symptoms_dataset.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Header: Disease, Symptoms... Severity, Duration
    header = ["disease"] + symptoms + ["severity", "duration"]
    writer.writerow(header)
    
    # Generate 12500 samples (500 per disease on average across 25 classes)
    for _ in range(12500):
        # Pick a random disease
        disease = random.choice(diseases)
        profile = disease_profiles[disease]
        
        # Determine symptoms (add some noise)
        row = [disease]
        for sym in symptoms:
            if sym in profile:
                # 85% chance a typical symptom is present
                row.append(1 if random.random() > 0.15 else 0)
            else:
                # 5% chance an unrelated symptom is present (noise)
                row.append(1 if random.random() > 0.95 else 0)
                
        # Severity (1: Low, 2: Medium, 3: High)
        severity = random.randint(1, 3)
        # Duration (1: Hours, 2: Days, 3: Weeks)
        duration = random.randint(1, 3)
        
        row.extend([severity, duration])
        
        writer.writerow(row)

print(f"Generated {filename} with 12500 samples across {len(diseases)} diseases and {len(symptoms)} symptoms.")
