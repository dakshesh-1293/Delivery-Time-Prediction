import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# -------------------------------
# LOAD SAVED FILES
# -------------------------------

@st.cache_resource
def load_files():

    model = joblib.load("delivery_xgboost_model.pkl")

    preprocessor = joblib.load("delivery_preprocessor.pkl")

    calibration = joblib.load("conformal_calibration.pkl")

    features = joblib.load("model_features.pkl")

    return model, preprocessor, calibration, features


try:
    model, preprocessor, calibration, features = load_files()

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# -------------------------------
# APPLICATION TITLE
# -------------------------------

st.title("🚚 Delivery Time Prediction System")

st.write(
    "Predict delivery duration using Machine Learning "
    "and Weather-wise Mondrian Conformal Prediction."
)

st.divider()


# -------------------------------
# INPUT FORM
# -------------------------------

st.subheader("📦 Enter Delivery Details")

col1, col2 = st.columns(2)

with st.form("delivery_form"):

    with col1:

        weekday = st.selectbox(
            "Weekday",
            ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
        )

        dispatch_hour = st.slider(
            "Dispatch Hour",
            7, 20, 12
        )

        distance_km = st.number_input(
            "Distance (KM)",
            min_value=0.59,
            max_value=65.0,
            value=10.0
        )

        package_weight_kg = st.number_input(
            "Package Weight (KG)",
            min_value=0.1,
            max_value=30.0,
            value=2.0
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["bike", "scooter", "van"]
        )

        traffic_level = st.selectbox(
            "Traffic Level",
            ["low", "medium", "high"]
        )

    with col2:

        weather = st.selectbox(
            "Weather Condition",
            ["clear", "cloudy", "rain", "storm"]
        )

        stop_sequence = st.slider(
            "Stop Sequence",
            1, 25, 1
        )

        driver_experience_months = st.number_input(
            "Driver Experience (Months)",
            min_value=0,
            max_value=120,
            value=12
        )

        warehouse_load_pct = st.slider(
            "Warehouse Load (%)",
            35.0, 100.0, 70.0
        )

        urban_density = st.selectbox(
            "Urban Density",
            ["rural", "suburban", "urban_core"]
        )

    submit = st.form_submit_button(
        "🚀 Predict Delivery Time",
        use_container_width=True
    )


# -------------------------------
# PREDICTION
# -------------------------------

if submit:

    input_data = pd.DataFrame([{

        "weekday": weekday,
        "dispatch_hour": dispatch_hour,
        "distance_km": distance_km,
        "package_weight_kg": package_weight_kg,
        "vehicle_type": vehicle_type,
        "traffic_level": traffic_level,
        "weather": weather,
        "stop_sequence": stop_sequence,
        "driver_experience_months": driver_experience_months,
        "warehouse_load_pct": warehouse_load_pct,
        "urban_density": urban_density

    }])

    # Match training feature order
    input_data = input_data[features]

    try:

        # Handle both Pipeline and standalone model
        if hasattr(model, "named_steps"):

            prediction = float(model.predict(input_data)[0])

        else:

            processed_data = preprocessor.transform(input_data)

            prediction = float(
                model.predict(processed_data)[0]
            )

        # Weather-wise calibration threshold
        if isinstance(calibration, dict):

            q = calibration.get(weather)

            if q is None:
                st.error("Calibration value not found for this weather.")
                st.stop()

        else:

            q = float(calibration)

        # Prediction interval
        lower_bound = max(0, prediction - q)

        upper_bound = prediction + q

        # -------------------------------
        # DISPLAY RESULTS
        # -------------------------------

        st.divider()

        st.subheader("📊 Prediction Results")

        result1, result2, result3 = st.columns(3)

        with result1:
            st.metric(
                "Predicted Delivery Time",
                f"{prediction:.2f} min"
            )

        with result2:
            st.metric(
                "Lower Bound",
                f"{lower_bound:.2f} min"
            )

        with result3:
            st.metric(
                "Upper Bound",
                f"{upper_bound:.2f} min"
            )

        st.success(
            f"Estimated delivery time: {prediction:.2f} minutes"
        )

        st.info(
            f"90% Prediction Interval: "
            f"{lower_bound:.2f} – {upper_bound:.2f} minutes"
        )

        st.write(
            f"**Weather Condition:** {weather.title()}"
        )

        st.write(
            f"**Weather Calibration Threshold:** {q:.2f} minutes"
        )

        st.caption(
            "The prediction interval is based on "
            "weather-wise Mondrian Conformal Prediction. "
            "The 90% coverage is a statistical target, "
            "not a guarantee for an individual delivery."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Developed by Dakshesh Parmar | "
    "M.Sc. Data Science | "
    "XGBoost + Mondrian Conformal Prediction"
)
