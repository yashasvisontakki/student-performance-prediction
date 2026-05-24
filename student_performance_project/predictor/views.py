"""
views.py  –  All the logic for Student Performance Predictor
"""
import os
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import PredictionHistory

# ── Paths to saved ML artefacts ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "ml_models")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
LE_PATH       = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODELS_PATH   = os.path.join(MODEL_DIR, "all_models.pkl")
RESULTS_PATH  = os.path.join(MODEL_DIR, "model_results.pkl")
FEATURE_NAMES = [
    "study_hours", "sleep_hours", "social_media_usage", "screen_time",
    "diet_quality", "mental_health", "physical_activity", "attendance",
    "stress_level", "internet_usage", "extra_curricular", "time_management",
]


def _models_ready():
    """Check if trained model files exist."""
    return all(os.path.exists(p) for p in [SCALER_PATH, LE_PATH, MODELS_PATH, RESULTS_PATH])


def _load_model_artifacts():
    """Load the trained scaler, encoder, models, and result metrics."""
    if not _models_ready():
        raise FileNotFoundError("Trained ML artefacts are not available.")

    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LE_PATH)
    models = joblib.load(MODELS_PATH)
    results = joblib.load(RESULTS_PATH)
    return scaler, le, models, results


def _models_ready():
    """Check if trained model files exist."""
    return all(os.path.exists(p) for p in [SCALER_PATH, LE_PATH, MODELS_PATH, RESULTS_PATH])


# ── Auth Views ────────────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('predict')

    if request.method == "POST":
        username  = request.POST.get("username", "").strip()
        email     = request.POST.get("email", "").strip()
        password1 = request.POST.get("password1", "")
        password2 = request.POST.get("password2", "")

        if not username or not password1:
            messages.error(request, "Username and password are required.")
        elif password1 != password2:
            messages.error(request, "Passwords do not match.")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Choose another.")
        elif len(password1) < 6:
            messages.error(request, "Password must be at least 6 characters.")
        else:
            User.objects.create_user(username=username, email=email, password=password1)
            messages.success(request, "Account created! Please log in.")
            return redirect("login")

    return render(request, "predictor/register.html")


def login_view(request):
    if request.user.is_authenticated:
        return redirect('predict')

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user     = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("predict")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "predictor/login.html")


def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("login")


# ── Main Prediction View ──────────────────────────────────────────────────────

@login_required
def predict_view(request):
    if not _models_ready():
        messages.error(
            request,
            "⚠️  ML models not found. Please run:  python train_models.py  first!"
        )
        return render(request, "predictor/predict.html", {"models_ready": False})

    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.POST.get("study_hours",        0)),
                float(request.POST.get("sleep_hours",        0)),
                float(request.POST.get("social_media_usage", 0)),
                float(request.POST.get("screen_time",        0)),
                float(request.POST.get("diet_quality",       0)),
                float(request.POST.get("mental_health",      0)),
                float(request.POST.get("physical_activity",  0)),
                float(request.POST.get("attendance",         0)),
                float(request.POST.get("stress_level",       0)),
                float(request.POST.get("internet_usage",     0)),
                float(request.POST.get("extra_curricular",   0)),
                float(request.POST.get("time_management",    0)),
            ]

            # Load artefacts
            scaler, le, models, results = _load_model_artifacts()

            # Scale input with explicit feature names to avoid sklearn warnings
            X = pd.DataFrame([features], columns=FEATURE_NAMES)
            X_scaled = scaler.transform(X)

            # Predict with every model
            predictions = {}
            for name, model in models.items():
                pred_enc = model.predict(X_scaled)[0]
                pred_label = le.inverse_transform([pred_enc])[0]
                predictions[name] = {
                    "label"   : pred_label,
                    "accuracy": float(results[name].get("accuracy", 0.0)),
                }

            # Best model by accuracy
            best_model = max(results.items(), key=lambda item: item[1].get("accuracy", 0.0))[0]
            final_label = predictions[best_model]["label"]
            best_accuracy = float(results[best_model].get("accuracy", 0.0))

            # Save to history
            PredictionHistory.objects.create(
                user              = request.user,
                study_hours       = features[0],
                sleep_hours       = features[1],
                social_media      = features[2],
                screen_time       = features[3],
                diet_quality      = int(features[4]),
                mental_health     = int(features[5]),
                physical_activity = features[6],
                attendance        = features[7],
                stress_level      = int(features[8]),
                internet_usage    = features[9],
                extra_curricular  = int(features[10]),
                time_management   = features[11],
                predicted_label   = final_label,
                best_model        = best_model,
            )

            # Store in session for results page
            request.session["predictions"] = predictions
            request.session["final_label"] = final_label
            request.session["best_model"] = best_model
            request.session["best_model_accuracy"] = best_accuracy
            request.session["input_data"] = dict(zip(
                ["Study Hours","Sleep Hours","Social Media Usage","Screen Time",
                 "Diet Quality","Mental Health","Physical Activity","Attendance",
                 "Stress Level","Internet Usage","Extra-Curricular","Time Management"],
                features
            ))
            return redirect("results")

        except FileNotFoundError as fnf_err:
            messages.error(request, f"Model files missing: {fnf_err}")
        except Exception as e:
            messages.error(request, f"Prediction error: {e}")

    return render(request, "predictor/predict.html", {"models_ready": True})


@login_required
def results_view(request):
    predictions = request.session.get("predictions")
    if not predictions:
        return redirect("predict")

    final_label = request.session.get("final_label", "Unknown")
    best_model = request.session.get("best_model", "Unknown")
    best_model_accuracy = request.session.get("best_model_accuracy", None)
    input_data = request.session.get("input_data", {})

    # Colour coding for label badges
    label_color = {"High": "success", "Medium": "warning", "Low": "danger"}.get(final_label, "secondary")

    context = {
        "predictions"          : predictions,
        "final_label"          : final_label,
        "best_model"           : best_model,
        "best_model_accuracy"  : best_model_accuracy,
        "input_data"           : input_data,
        "label_color"          : label_color,
    }
    return render(request, "predictor/results.html", context)


@login_required
def history_view(request):
    history = PredictionHistory.objects.filter(user=request.user)[:20]
    return render(request, "predictor/history.html", {"history": history})
