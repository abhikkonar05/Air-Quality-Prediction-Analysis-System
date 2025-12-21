# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # ✅ CHANGED: Much faster than pickle for ML models
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ===============================
# 2. LOAD DATA
# ===============================
# ✅ Added 'r' for raw string to fix path errors
data = pd.read_csv(r"C:\Users\hrida\Documents\machine learning projects\Air-Quality-Prediction-Analysis-System\Air_Quality_Purple_Air_Sensors.csv")
df = pd.DataFrame(data)

# ===============================
# 3. HANDLE MISSING VALUES
# ===============================
# Fast imputation
fill_means = ['HUMIDITY', 'PM1', 'PM2_5', 'PM10', 'PM2_5_CF_1', 'PM2_5_ALT']
fill_medians = ['TEMPERATURE', 'PRESSURE', 'CONFIDENCE', 'VOC']

for col in fill_means:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

for col in fill_medians:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ===============================
# 4. FEATURE ENGINEERING
# ===============================
df['REPORTED_DATETIME'] = pd.to_datetime(df['REPORTED_DATETIME'])

df['HOUR'] = df['REPORTED_DATETIME'].dt.hour
df['MONTH'] = df['REPORTED_DATETIME'].dt.month
df['DAY_OF_WEEK'] = df['REPORTED_DATETIME'].dt.dayofweek
df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
df['LAT_ROUND'] = df['LATITUDE'].round(2)
df['LON_ROUND'] = df['LONGITUDE'].round(2)

# ⚠️ NOTE: Including PM1 and PM10 here causes 100% accuracy because they are too similar to PM2.5
FEATURES = [
    'HUMIDITY', 'TEMPERATURE', 'PRESSURE',
    'PM1', 'PM10', 'VOC',
    'HOUR', 'MONTH', 'DAY_OF_WEEK',
    'IS_WEEKEND', 'LAT_ROUND', 'LON_ROUND'
]

X = df[FEATURES]

# ===============================
# 5. REGRESSION TASK (PM2.5)
# ===============================
print("\n--- STARTING REGRESSION TRAINING ---")
y_reg = df['PM2_5']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ✅ OPTIMIZED MODELS
regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5, n_jobs=-1), # Parallel
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    # Reduced n_estimators to 100 and added max_depth to reduce file size
    "Random Forest Regressor": RandomForestRegressor(n_jobs=-1, verbose=1, n_estimators=100, max_depth=20, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

reg_results = {}
trained_reg_models = {}

for name, model in regression_models.items():
    print(f"Training {name}...")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    reg_results[name] = [mae, rmse]
    trained_reg_models[name] = model

reg_results_df = pd.DataFrame(reg_results, index=["MAE", "RMSE"]).T
print("\nRegression Summary:")
print(reg_results_df)

# ===============================
# 6. CLASSIFICATION TASK
# ===============================
print("\n--- STARTING CLASSIFICATION TRAINING ---")
df['AIR_QUALITY_LABEL'] = (df['PM2_5'] > 50).astype(int)
y_clf = df['AIR_QUALITY_LABEL']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# Use the SAME scaler logic to be consistent
X_train_c_s = scaler.fit_transform(X_train_c)
X_test_c_s = scaler.transform(X_test_c)

classification_models = {
    "Logistic Regression": LogisticRegression(n_jobs=-1),
    "KNN Classifier": KNeighborsClassifier(n_jobs=-1),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    # ✅ OPTIMIZED: n_jobs=-1 for speed, max_depth=20 for file size
    "Random Forest Classifier": RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=20, random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    "AdaBoost Classifier": AdaBoostClassifier(random_state=42)
}

class_results = {}
trained_class_models = {}

for name, model in classification_models.items():
    print(f"Training {name}...")
    model.fit(X_train_c_s, y_train_c)
    preds = model.predict(X_test_c_s)

    acc = accuracy_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)

    class_results[name] = [acc, f1]
    trained_class_models[name] = model

class_results_df = pd.DataFrame(class_results, index=["Accuracy", "F1 Score"]).T
print("\nClassification Summary:")
print(class_results_df)

# ===============================
# 7. VISUALIZE CONFUSION MATRIX
# ===============================
best_clf_name = "Random Forest Classifier"
best_classifier = trained_class_models[best_clf_name]
pred_best = best_classifier.predict(X_test_c_s)

cm = confusion_matrix(y_test_c, pred_best)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_clf_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 8. SAVE MODELS (JOBLIB)
# ===============================
print("\n--- SAVING MODELS ---")

# Using compress=3 reduces file size significantly (1-9, higher is smaller but slower)
joblib.dump(trained_reg_models, "regression_models.joblib", compress=3)
joblib.dump(trained_class_models, "classification_models.joblib", compress=3)
joblib.dump(scaler, "scaler.joblib")

print("✅ Models saved successfully using Joblib (Compressed)!")