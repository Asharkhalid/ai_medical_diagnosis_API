import os
from datetime import datetime

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

import models
from database import engine, SessionLocal

# Idempotent: this script can run standalone (before the API has ever
# started) as well as via the Admin Console's retrain button.
models.Base.metadata.create_all(bind=engine)

print("Loading dataset...")
df = pd.read_csv('symptoms_dataset.csv')

X = df.drop('disease', axis=1)
y = df['disease']

# Save the feature columns so the web app can use them exactly
feature_columns = list(X.columns)
os.makedirs('models', exist_ok=True)
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FR-07: the system shall support Decision Tree, Naive Bayes, and Random
# Forest algorithms. All three are trained on the same split so their
# metrics are directly comparable in the Admin "Model Performance" tab.
ALGORITHMS = {
    "random_forest": ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    "decision_tree": ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    "naive_bayes": ("GaussianNB", GaussianNB()),
}

trained_at = datetime.utcnow()
version = trained_at.strftime("v%Y.%m.%d")

db = SessionLocal()
try:
    any_active_exists = db.query(models.ModelMetadata).filter(models.ModelMetadata.is_active.is_(True)).first() is not None

    for algo_key, (display_name, clf) in ALGORITHMS.items():
        print(f"Training {display_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        print(f"{display_name} -> accuracy: {accuracy:.2f}%, precision: {precision:.2f}%, recall: {recall:.2f}%")

        joblib.dump(clf, f'models/{algo_key}_v1.pkl')

        row = db.query(models.ModelMetadata).filter(models.ModelMetadata.algorithm == algo_key).first()
        if row:
            row.model_version = version
            row.trained_at = trained_at
            row.accuracy = accuracy
            row.precision_score = precision
            row.recall_score = recall
        else:
            db.add(models.ModelMetadata(
                model_name=display_name,
                model_version=version,
                algorithm=algo_key,
                trained_at=trained_at,
                accuracy=accuracy,
                precision_score=precision,
                recall_score=recall,
                # First-ever training run: default Random Forest to active
                # so live predictions keep working without an extra admin
                # step. Later runs never change which model is active.
                is_active=(algo_key == "random_forest" and not any_active_exists),
            ))

    db.commit()
finally:
    db.close()

print("Training Complete!")
