"""
train_models.py
Run this ONCE before starting the Django server.
It creates a sample dataset and trains all 6 ML models.

Usage:  python train_models.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ── 1. Generate a realistic sample dataset ──────────────────────────────────
np.random.seed(42)
N = 1000

study_hours       = np.random.uniform(1, 10, N)
sleep_hours       = np.random.uniform(4, 10, N)
social_media      = np.random.uniform(0, 8, N)
screen_time       = np.random.uniform(1, 12, N)
diet_quality      = np.random.randint(1, 6, N)        # 1-5 scale
mental_health     = np.random.randint(1, 11, N)       # 1-10 scale
physical_activity = np.random.uniform(0, 5, N)        # hours/week
attendance        = np.random.uniform(50, 100, N)     # percentage
stress_level      = np.random.randint(1, 11, N)       # 1-10 scale
internet_usage    = np.random.uniform(1, 10, N)
extra_curricular  = np.random.randint(0, 2, N)        # 0 or 1
time_management   = np.random.uniform(1, 10, N)

# Create performance score (weighted combination)
score = (
    study_hours       * 4.0 +
    sleep_hours       * 2.0 +
    (10 - social_media) * 1.5 +
    diet_quality      * 2.0 +
    mental_health     * 2.0 +
    physical_activity * 1.5 +
    attendance        * 0.3 +
    (10 - stress_level) * 1.5 +
    time_management   * 2.0 +
    extra_curricular  * 3.0
)

# Assign labels based on percentile
low_thresh  = np.percentile(score, 33)
high_thresh = np.percentile(score, 66)

def assign_label(s):
    if s < low_thresh:
        return "Low"
    elif s < high_thresh:
        return "Medium"
    else:
        return "High"

labels = np.array([assign_label(s) for s in score])

df = pd.DataFrame({
    "study_hours"       : study_hours,
    "sleep_hours"       : sleep_hours,
    "social_media_usage": social_media,
    "screen_time"       : screen_time,
    "diet_quality"      : diet_quality,
    "mental_health"     : mental_health,
    "physical_activity" : physical_activity,
    "attendance"        : attendance,
    "stress_level"      : stress_level,
    "internet_usage"    : internet_usage,
    "extra_curricular"  : extra_curricular,
    "time_management"   : time_management,
    "performance_label" : labels,
})

os.makedirs("predictor/ml_models", exist_ok=True)
df.to_csv("predictor/ml_models/dataset.csv", index=False)
print("✔  Dataset saved → predictor/ml_models/dataset.csv")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
X = df.drop("performance_label", axis=1)
y = df["performance_label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)          # High=0, Low=1, Medium=2 (alphabetical)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. Train all 6 models ─────────────────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                 : SVC(kernel="rbf", probability=True, random_state=42),
    "KNN"                 : KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes"         : GaussianNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds   = model.predict(X_test_sc)
    acc     = accuracy_score(y_test, preds)
    cm      = confusion_matrix(y_test, preds).tolist()
    results[name] = {"accuracy": round(acc * 100, 2), "confusion_matrix": cm}
    print(f"  {name:22s}  Accuracy = {acc*100:.2f}%")

# ── 4. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(scaler,  "predictor/ml_models/scaler.pkl")
joblib.dump(le,      "predictor/ml_models/label_encoder.pkl")
joblib.dump(models,  "predictor/ml_models/all_models.pkl")
joblib.dump(results, "predictor/ml_models/model_results.pkl")

print("\n✔  All models & artefacts saved to predictor/ml_models/")
print("✔  Training complete!  Now run:  python manage.py runserver")
