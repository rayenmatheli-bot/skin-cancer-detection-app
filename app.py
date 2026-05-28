from flask import Flask, render_template, request, redirect, session, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("model\vgg16_skin_cancer.h5")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="skin_cancer_db"
)
cursor = db.cursor(dictionary=True)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (user, pwd))
        result = cursor.fetchone()

        if result:
            session["user"] = user
            flash("Login réussi ✔", "success")
            return redirect("/dashboard")
        else:
            flash("Erreur login ❌", "danger")

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        try:
            name = request.form["name"]
            age = request.form["age"]
            file = request.files["image"]

            if file.filename == "":
                flash("Veuillez choisir une image", "warning")
                return redirect("/predict")

            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            img = image.load_img(path, target_size=(224, 224))
            h = image.img_to_array(img) / 255.0
            img = np.expand_dims(h, axis=0)
            
            preds = model.predict(img)[0]
            
            # Correction de la logique d'extraction et de condition
            if len(preds) > 1:
                # Si le modèle est catégorique (2 classes)
                malignant_prob = preds[1] 
                benign_prob = preds[0]
                pred = malignant_prob
            else:
                # Si le modèle est binaire (1 seule sortie)
                pred = preds[0]
                malignant_prob = pred
                benign_prob = 1 - pred
                
            result = "Malignant" if pred > 0.5 else "Benign"
            display_prob = pred if result == "Malignant" else benign_prob

            cursor.execute("""
                INSERT INTO patients (name, age, result, probability, image_path)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, age, result, float(display_prob), path))
            db.commit()
            
            flash("Analyse réussie ✔", "success")
            return render_template("result.html", result=result, prob=round(display_prob * 100, 2), img=path)
        except Exception as e:
            
            flash(f"Erreur système ❌: {str(e)}", "danger")
            return redirect("/predict")
            
    return render_template("predict.html")

@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect("/")
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    data = cursor.fetchall()
    return render_template("patients.html", patients=data)

@app.route("/logout")
def logout():
    session.clear()
    flash("Déconnecté", "info")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
