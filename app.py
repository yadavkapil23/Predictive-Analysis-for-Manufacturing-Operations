from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


uploaded_data = None
logistic_model = None
decision_tree_model = None


#upload end
@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_data
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded."}), 400


    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # load the CSV or execl file
    if file.filename.endswith(".csv"):
        uploaded_data = pd.read_csv(file_path)
    elif file.filename.endswith((".xlsx", ".xls")):
        uploaded_data = pd.read_excel(file_path)
    else:
        return jsonify({"error": "Only CSV or Excel files are supported."}), 400

    return jsonify({"message": "File uploaded successfully", "columns": list(uploaded_data.columns)})


# Train Model endpt.
@app.route("/train", methods=["POST"])
def train_models():
    global uploaded_data, logistic_model, decision_tree_model

    if uploaded_data is None:
        return jsonify({"error": "No data uploaded. Use /upload first."}), 400

    if "DefectStatus" not in uploaded_data.columns:
        return jsonify({"error": "Dataset must include a 'DefectStatus' column."}), 400


    X = uploaded_data.drop(columns=["DefectStatus"])
    Y = uploaded_data["DefectStatus"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, Y_train)
    logistic_preds = logistic_model.predict(X_test)

    # Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, Y_train)
    decision_tree_preds = decision_tree_model.predict(X_test)


    logistic_metrics = {
        "accuracy": accuracy_score(Y_test, logistic_preds),
        "f1_score": f1_score(Y_test, logistic_preds),
    }
    decision_tree_metrics = {
        "accuracy": accuracy_score(Y_test, decision_tree_preds),
        "f1_score": f1_score(Y_test, decision_tree_preds),
    }

    # Save models
    joblib.dump(logistic_model, "logistic_model.pkl")
    joblib.dump(decision_tree_model, "decision_tree_model.pkl")

    return jsonify({
        "logistic_regression": logistic_metrics,
        "decision_tree": decision_tree_metrics,
    })



@app.route("/predict", methods=["POST"])
def predict():
    global logistic_model, decision_tree_model

    data = request.get_json()
    model_type = data.get("model_type", "logistic")

    input_columns = ["ProductionVolume", "ProductionCost", "SupplierQuality", "DeliveryDelay",
                     "DefectRate", "QualityScore", "MaintenanceHours", "DowntimePercentage", 
                     "InventoryTurnover", "StockoutRate", "WorkerProductivity", "SafetyIncidents", 
                     "EnergyConsumption", "EnergyEfficiency", "AdditiveProcessTime", "AdditiveMaterialCost"]


    input_features = pd.DataFrame([data], columns=input_columns)

    if model_type == "logistic" and logistic_model is None:
        return jsonify({"error": "Logistic Regression model not trained."}), 400
    if model_type == "decision_tree" and decision_tree_model is None:
        return jsonify({"error": "Decision Tree model not trained."}), 400


    model = logistic_model if model_type == "logistic" else decision_tree_model
    prediction = model.predict(input_features)


    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": round(model.predict_proba(input_features)[0][1], 2)  
    })


if __name__ == "__main__":
    app.run(debug=True)
