

from flask import (
    Flask, jsonify, request, send_from_directory, session
)
import os
import sqlite3
import math
from datetime import datetime, timezone
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FOLDER = os.path.abspath(
    os.path.join(BACKEND_FOLDER, "..", "frontend")
)
DATABASE_PATH = os.path.join(BACKEND_FOLDER, "student_predictions.db")

# Set a strong SECRET_KEY environment variable before deployment.
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "development-only-change-before-deployment"
)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Existing Linear Regression coefficients
MODEL_INTERCEPT = -24.730001779098856
STUDY_HOURS_COEF = 0.75657208
PREVIOUS_SCORE_COEF = 0.44199956
ATTENDANCE_COEF = 0.73989333


# Initialize database tables
def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_hours REAL NOT NULL,
                previous_score REAL NOT NULL,
                attendance REAL NOT NULL,
                predicted_score REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Read existing column names. PRAGMA returns tuples,
        # so row[1] is the column name.
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        # Add this column to older databases without deleting their data.
        if "user_id" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN user_id INTEGER"
            )


# Protect routes that require login
def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({
                "error": "Please log in to continue."
            }), 401

        return function(*args, **kwargs)

    return wrapper


@app.route("/")
def home():
    return send_from_directory(FRONTEND_FOLDER, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "backend": "online",
        "ml_service": "connected",
        "database": "enabled"
    })


# Register a new user
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send valid JSON data."
        }), 400

    username = data.get("username", "")
    password = data.get("password", "")

    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify({
            "error": "Username and password must be text."
        }), 400

    username = username.strip()

    if not 3 <= len(username) <= 30:
        return jsonify({
            "error": "Username must contain 3 to 30 characters."
        }), 400

    if len(password) < 8:
        return jsonify({
            "error": "Password must contain at least 8 characters."
        }), 400

    password_hash = generate_password_hash(password)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute("""
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
            """, (username, password_hash, created_at))
    except sqlite3.IntegrityError:
        return jsonify({
            "error": "Username already exists."
        }), 409

    return jsonify({
        "message": "Registration successful. You can now log in."
    }), 201


# Log in an existing user
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send valid JSON data."
        }), 400

    username = data.get("username", "")
    password = data.get("password", "")

    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify({
            "error": "Username and password must be text."
        }), 400

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        user = connection.execute("""
            SELECT id, username, password_hash
            FROM users
            WHERE username = ?
        """, (username.strip(),)).fetchone()

    if user is None or not check_password_hash(
        user["password_hash"], password
    ):
        return jsonify({
            "error": "Invalid username or password."
        }), 401

    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]

    return jsonify({
        "message": "Login successful.",
        "username": user["username"]
    })


# Log out
@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()

    return jsonify({
        "message": "Logged out successfully."
    })


# Check current login status
@app.route("/me", methods=["GET"])
def current_user():
    if "user_id" not in session:
        return jsonify({"logged_in": False})

    return jsonify({
        "logged_in": True,
        "user_id": session["user_id"],
        "username": session["username"]
    })


# Create a prediction for the logged-in user
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send valid JSON data."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        previous_score = float(data["previous_score"])
        attendance = float(data["attendance"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": (
                "Provide study_hours, previous_score, "
                "and attendance as numbers."
            )
        }), 400

    values = [study_hours, previous_score, attendance]

    if not all(math.isfinite(value) for value in values):
        return jsonify({
            "error": "Values must be finite numbers."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    if not 0 <= previous_score <= 100:
        return jsonify({
            "error": "Previous score must be between 0 and 100."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    predicted_score = (
        MODEL_INTERCEPT
        + STUDY_HOURS_COEF * study_hours
        + PREVIOUS_SCORE_COEF * previous_score
        + ATTENDANCE_COEF * attendance
    )

    predicted_score = max(0, min(100, predicted_score))
    predicted_score = round(predicted_score, 2)
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute("""
            INSERT INTO predictions (
                user_id,
                study_hours,
                previous_score,
                attendance,
                predicted_score,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            study_hours,
            previous_score,
            attendance,
            predicted_score,
            created_at
        ))

        prediction_id = cursor.lastrowid

    return jsonify({
        "id": prediction_id,
        "predicted_score": predicted_score,
        "model": "Linear Regression",
        "status": "success"
    })


# Retrieve the logged-in user's prediction history
@app.route("/predictions", methods=["GET"])
@login_required
def get_predictions():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT
                id,
                study_hours,
                previous_score,
                attendance,
                predicted_score,
                created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
        """, (session["user_id"],)).fetchall()

    return jsonify({
        "count": len(rows),
        "predictions": [dict(row) for row in rows]
    })


# Clear only the logged-in user's prediction history
@app.route("/predictions", methods=["DELETE"])
@login_required
def clear_predictions():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            DELETE FROM predictions
            WHERE user_id = ?
        """, (session["user_id"],))

    return jsonify({
        "message": "Your prediction history was cleared successfully."
    })


if __name__ == "__main__":
    initialize_database()
    app.run(debug=True)

