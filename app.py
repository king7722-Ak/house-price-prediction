import streamlit as st
import joblib

model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="Dream House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
/* ... (your same CSS here) ... */
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("""
        <div class="left-panel">
            <h1>🏡 Predict your Dream House Price</h1>
            <p>Get an instant estimate based on your selected features.</p>
            <button class="predict-button">Start Prediction</button>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("Enter House Details")

    area = st.number_input("Area (in sqft)", min_value=500, max_value=10000, value=1500)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=5)
    garage = st.selectbox("Garage Spaces", [0, 1, 2, 3])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    pool = st.selectbox("Swimming Pool", ["No", "Yes"])
    gym = st.selectbox("Gym", ["No", "Yes"])

    predict_button = st.button("Predict Price 🏠")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("Estimated Price")

    if predict_button:
        pool_val = 1 if pool == "Yes" else 0
        gym_val = 1 if gym == "Yes" else 0

        # Must match model feature order
        features = [[area, bedrooms, age, garage, bathrooms, pool_val, gym_val]]
        predicted_price = model.predict(features)[0]

        st.success(f"💰 Estimated Price: ₹ {predicted_price:,.2f}")
        st.caption("This is an approximate price based on your inputs.")
    else:
        st.info("Please enter house details and click Predict.")

    st.markdown('</div>', unsafe_allow_html=True)
