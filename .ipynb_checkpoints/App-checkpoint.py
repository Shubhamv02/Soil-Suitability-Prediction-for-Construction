import streamlit as st
import numpy as np
import pickle

# Load the model and encoders
model = pickle.load(open('soil_model.pkl', 'rb'))
scaler = pickle.load(open('soil_scaler.pkl', 'rb'))
label_encoder = pickle.load(open('soil_encoder.pkl', 'rb'))


# Page config
st.set_page_config(page_title="Soil Suitability App", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            color:#4CAF50;
            text-align:center;
        }
        .subtitle {
            font-size:20px;
            text-align:center;
            color:#666666;
        }
        .result-box {
            background-color:#f1f1f1;
            padding:20px;
            border-radius:10px;
            text-align:center;
            margin-top:20px;
        }
        .suitable {
            color:green;
            font-size:24px;
            font-weight:bold;
        }
        .not-suitable {
            color:red;
            font-size:24px;
            font-weight:bold;
        }
        .emoji {
            font-size:50px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">🏗️ Soil Suitability Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter soil details to check construction suitability</div>', unsafe_allow_html=True)
st.markdown("---")

# User input
ph = st.slider("Select Soil pH", 0.0, 14.0, 7.0, 0.1)
moisture = st.slider("Select Moisture %", 0.0, 100.0, 30.0, 0.1)
soil_type = st.selectbox("Choose Soil Type", ["Sandy", "Silty", "Peaty", "Loamy"])

if st.button("Predict"):
    try:
        soil_encoded = label_encoder.transform([soil_type])[0]
        input_data = np.array([[ph, moisture, soil_encoded]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            # Suitable message
            st.markdown("""
                <div class="result-box">
                    <div class="emoji">🎉</div>
                    <div class="suitable">The soil is Suitable for Construction!</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Not suitable message
            st.markdown("""
                <div class="result-box">
                    <div class="emoji">😞</div>
                    <div class="not-suitable">The soil is Not Suitable for Construction.</div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error occurred: {e}")

