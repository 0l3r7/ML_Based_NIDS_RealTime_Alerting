import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Network Intrusion Detection System")
uploaded_file = st.file_uploader("Upload Network Traffic CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file, header=None)
    label_column = data.columns[-2]

    X = pd.get_dummies(data.drop(label_column, axis=1))
    X = X.reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    data["prediction"] = predictions
    
    # Optional attack_type fallback
    if "attack_type" not in data.columns:
        data["attack_type"] = data[label_column].apply(lambda x: "Normal" if x==0 else "Attack")
    
    attacks = data[data["prediction"] == 1]
    normal = data[data["prediction"] == 0]

    st.write("Total traffic:", len(data))
    st.write("Detected attacks:", len(attacks))
    st.write("Normal traffic:", len(normal))

    if len(attacks) > 0:
        st.error("⚠ Intrusion Detected!")

    st.dataframe(data)

    # Pie chart
    traffic_counts = data["prediction"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(
        traffic_counts,
        labels=["Normal", "Attack"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#F44336"]
    )
    ax1.axis("equal")
    st.pyplot(fig1)

    # Bar chart
    attack_types = data[data["prediction"] == 1]["attack_type"].value_counts()
    fig2, ax2 = plt.subplots()
    attack_types.plot(kind="bar", ax=ax2, color="#F44336")
    ax2.set_title("Attack Types Detected")
    ax2.set_xlabel("Attack Type")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)