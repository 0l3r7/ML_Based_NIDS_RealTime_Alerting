import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("dataset/network.csv", header=None)
label_column = data.columns[-2]
data[label_column] = data[label_column].apply(lambda x: 0 if x=="normal" else 1)

# Features & target
X = pd.get_dummies(data.drop(label_column, axis=1))
X.columns = X.columns.astype(str)
y = data[label_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, pred))

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns, "columns.pkl")
print("Model, scaler, and columns saved!")