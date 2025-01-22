Manufacturing Operations Prediction API


 Introduction
 This API allows users to upload manufacturing data, train machine learning models, and make
 predictions about machine downtime or defects. The project uses Flask for the API and scikit-learn
 for modeling.

 
 Setup Instructions
 1. Install dependencies:
   pip install flask pandas scikit-learn joblib

 2. Run the Flask application:
   python app.py

 3. The API will be accessible at:
   http://127.0.0.1:5000/



 API Endpoints
 1. Upload Dataset (POST /upload)- Description: Uploads a CSV or Excel file containing manufacturing data.- Request: Send a file with key columns like ProductionVolume, DefectRate, etc.- Example cURL:
  curl -X POST -F "file=@manufacturing_data.csv" http://127.0.0.1:5000/upload


 2. Train Model (POST /train)- Description: Trains the model using uploaded data and returns performance metrics.-
    Example Response:
  {

  }
  
    "logistic_regression": {"accuracy": 0.87, "f1_score": 0.93},
    "decision_tree": {"accuracy": 0.90, "f1_score": 0.94}

    
 3. Predict Outcome (POST /predict)

    
- Description:

Accepts JSON input and returns predictions.- 
Example Request:
  {
  }
  
    "model_type": "logistic",
     "ProductionVolume": 1200,
    "ProductionCost": 50000,
    "SupplierQuality": 8,
    "DeliveryDelay": 1,
    "DefectRate": 0.02,
    "QualityScore": 95,
    "MaintenanceHours": 120,
    "DowntimePercentage": 10,
    "InventoryTurnover": 5,
    "StockoutRate": 0.1,
    "WorkerProductivity": 90,
    "SafetyIncidents": 1,
    "EnergyConsumption": 1000,
    "EnergyEfficiency": 0.85,
    "AdditiveProcessTime": 5,
    "AdditiveMaterialCost": 200
}     -  Example Response:
  {
  }

  OUTPUT : 
  
    "Downtime": "Yes",
    "Confidence": 0.90
