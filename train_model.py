import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('symptoms_dataset.csv')

X = df.drop('disease', axis=1)
y = df['disease']

# Save the feature columns so the web app can use them exactly
feature_columns = list(X.columns)
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("Saving model to disk...")
import os
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/random_forest_v1.pkl')
print("Training Complete!")
