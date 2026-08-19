import csv
import random

# Target diseases
diseases = [
    "Common Cold", "Influenza", "COVID-19", "Acute Migraine",
    "Gastroenteritis", "Dengue Fever", "Typhoid", "Pneumonia",
    "Bronchitis", "Tuberculosis", "Malaria", "Asthma",
    "Anemia", "Diabetes Type 2", "Hypertension"
]

# All possible symptoms across the diseases
symptoms = [
    "fever", "headache", "cough", "fatigue", "abdominal_pain",
    "nausea", "vomiting", "sore_throat", "shortness_of_breath",
    "body_pain", "chills", "loss_of_taste", "loss_of_smell",
    "diarrhea", "chest_pain", "wheezing", "dizziness",
    "weight_loss", "sweating", "pale_skin", "frequent_urination"
]

# Disease symptom profiles (which symptoms they typically present)
disease_profiles = {
    "Common Cold": ["cough", "sore_throat", "fatigue", "body_pain"],
    "Influenza": ["fever", "chills", "body_pain", "fatigue", "headache", "cough"],
    "COVID-19": ["fever", "cough", "loss_of_taste", "loss_of_smell", "fatigue", "shortness_of_breath"],
    "Acute Migraine": ["headache", "nausea", "dizziness", "vomiting"],
    "Gastroenteritis": ["abdominal_pain", "nausea", "vomiting", "diarrhea", "fever"],
    "Dengue Fever": ["fever", "headache", "body_pain", "fatigue", "nausea"],
    "Typhoid": ["fever", "headache", "abdominal_pain", "fatigue", "weight_loss"],
    "Pneumonia": ["cough", "fever", "chills", "shortness_of_breath", "chest_pain"],
    "Bronchitis": ["cough", "fatigue", "shortness_of_breath", "chest_pain"],
    "Tuberculosis": ["cough", "weight_loss", "fatigue", "fever", "sweating"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea", "body_pain"],
    "Asthma": ["shortness_of_breath", "chest_pain", "wheezing", "cough"],
    "Anemia": ["fatigue", "pale_skin", "dizziness", "chest_pain", "shortness_of_breath"],
    "Diabetes Type 2": ["frequent_urination", "fatigue", "weight_loss"],
    "Hypertension": ["headache", "dizziness", "fatigue", "chest_pain"]
}

filename = 'symptoms_dataset.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Header: Disease, Symptoms... Severity, Duration
    header = ["disease"] + symptoms + ["severity", "duration"]
    writer.writerow(header)
    
    # Generate 5000 samples
    for _ in range(5000):
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

print(f"Generated {filename} with 5000 samples.")
