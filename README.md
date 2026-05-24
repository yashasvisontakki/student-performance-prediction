# 🎓 Student Performance Prediction
### Machine Learning + Django — Karunadu Technologies Internship Project

---

## 📁 Project Structure

```
student_performance_project/
│
├── manage.py                        ← Django manager (run server from here)
├── requirements.txt                 ← All Python packages needed
├── train_models.py                  ← Run ONCE to train ML models
│
├── student_performance/             ← Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── predictor/                       ← Main Django app
    ├── models.py                    ← Database models (PredictionHistory)
    ├── views.py                     ← All page logic
    ├── urls.py                      ← URL routes
    ├── apps.py
    ├── migrations/
    ├── ml_models/                   ← Created after training
    │   ├── dataset.csv
    │   ├── scaler.pkl
    │   ├── label_encoder.pkl
    │   ├── all_models.pkl
    │   └── model_results.pkl
    └── templates/predictor/
        ├── base.html                ← Common layout (navbar, footer)
        ├── login.html               ← Login page
        ├── register.html            ← Registration page
        ├── predict.html             ← Input form page
        ├── results.html             ← Prediction results page
        └── history.html             ← User's past predictions
```

---

## 🚀 Setup Instructions (Step-by-Step for Beginners)

### Step 1 — Install Python
Make sure Python 3.9+ is installed.
Check: `python --version`

### Step 2 — Open Terminal / Command Prompt
Navigate to the project folder:
```
cd student_performance_project
```

### Step 3 — Install Required Packages
```
pip install -r requirements.txt
```
This installs Django, scikit-learn, pandas, numpy, joblib.

### Step 4 — Train the ML Models (Run ONCE)
```
python train_models.py
```
This will:
- Generate a sample dataset (1000 students)
- Train 6 classification models
- Save all models inside `predictor/ml_models/`
- Print accuracy of each model

### Step 5 — Set Up the Database
```
python manage.py makemigrations
'python manage.py migrate'
```

### Step 6 — Create Admin Account (Optional)
```
python manage.py createsuperuser
```

### Step 7 — Start the Web Server
```

python mapython manage.py createsuperuser

```

### Step 8 — Open in Browser
Go to: http://127.0.0.1:8000

---

## 🌐 Pages in the App

| URL | Page | Description |
|-----|------|-------------|
| `/register/` | Register | Create a new account |
| `/login/` | Login | Sign into your account |
| `/predict/` | Predict | Fill in habits and get prediction |
| `/results/` | Results | See prediction + model comparison |
| `/history/` | History | View past 20 predictions |
| `/admin/` | Admin | Django admin panel |

---

## 🤖 ML Models Used

| Model | Type |
|-------|------|
| Logistic Regression | Linear classifier |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of trees |
| SVM (Support Vector Machine) | Margin-based classifier |
| KNN (K-Nearest Neighbors) | Distance-based classifier |
| Naive Bayes | Probabilistic classifier |

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| Study Hours | Hours studied per day |
| Sleep Hours | Hours slept per night |
| Social Media Usage | Hours on social media per day |
| Screen Time | Total screen time per day |
| Diet Quality | Scale 1–5 |
| Mental Health | Scale 1–10 |
| Physical Activity | Hours of exercise per week |
| Attendance | Percentage attendance |
| Stress Level | Scale 1–10 |
| Internet Usage | Hours online per day |
| Extra-Curricular | 0 = No, 1 = Yes |
| Time Management | Scale 1–10 |

**Output (Performance Label):** `High`, `Medium`, or `Low`

---

## ❓ Interview Questions & Answers

**Q: What is classification vs regression?**
A: Classification predicts a category (e.g., High/Medium/Low). Regression predicts a number (e.g., exact marks).

**Q: What is overfitting?**
A: When a model learns training data too well but fails on new data. Fix: use cross-validation, pruning, or more data.

**Q: What is a Confusion Matrix?**
A: A table showing True Positives, False Positives, True Negatives, False Negatives — used to evaluate model quality.

**Q: What does Random Forest improve over Decision Tree?**
A: Random Forest trains many trees on random subsets, reducing overfitting and improving accuracy.

**Q: What are Django views, models, templates?**
A: Models = database tables. Views = logic/functions. Templates = HTML pages shown to users.

---

## 📞 Contact
Karunadu Technologies Private Limited
Email: karunadutechnologies@gmail.com
Web: www.karunadutechnologies.com
