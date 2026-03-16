import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("dataset/network.csv", header=None)
label_column = data.columns[-2]
data[label_column] = data[label_column].apply(lambda x: 0 if x=="normal" else 1)

# ----------------------------
# Features & target
# ----------------------------
X = pd.get_dummies(data.drop(label_column, axis=1))
X.columns = X.columns.astype(str)
y = data[label_column]

# ----------------------------
# Train-test split for Random Forest
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Random Forest
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Evaluate RF
pred = rf_model.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# ----------------------------
# Train Isolation Forest on benign traffic
# ----------------------------
X_benign = X[y == 0]
iso_scaler = StandardScaler()
X_benign_scaled = iso_scaler.fit_transform(X_benign)

iso_model = IsolationForest(n_estimators=500, contamination=0.01, random_state=42)
iso_model.fit(X_benign_scaled)

# ----------------------------
# Save models and objects
# ----------------------------
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns, "columns.pkl")

joblib.dump(iso_model, "iso_model.pkl")
joblib.dump(iso_scaler, "iso_scaler.pkl")

print("All models saved: rf_model.pkl, iso_model.pkl, scalers, and columns.pkl")