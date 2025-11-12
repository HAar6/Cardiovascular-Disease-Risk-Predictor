import os
import streamlit as st
from backend import model as model_module
import datetime
import io


# Startup ONNX health-check: try to load the ONNX model once at app start and
# display a friendly error if loading fails. Uses `onnxruntime` when available
# otherwise falls back to the pure-Python loader in `backend.model`.
@st.cache_resource
def _load_onnx_health():
    try:
        onnx_path = getattr(model_module, 'ONNX_PATH', None)
        if not onnx_path or not os.path.exists(onnx_path):
            return (False, f"ONNX model not found at {onnx_path}. Ensure `backend/binary_model.onnx` is present in the repo.")

        # Prefer onnxruntime if it's installed (local dev). If not, use the
        # pure-Python loader implemented in `backend.model`.
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            # Basic smoke run with zeros to ensure model executes
            import numpy as _np
            inp_name = sess.get_inputs()[0].name
            sess.run(None, {inp_name: _np.zeros((1, 13), dtype=_np.float32)})
            return (True, 'onnxruntime')
        except Exception:
            # Fall back to pure-Python loader from backend.model
            try:
                loader = getattr(model_module, '_load_onnx_pure_python', None)
                if loader is None:
                    return (False, 'No fallback ONNX loader available in backend.model')
                f = loader(onnx_path)
                # Run a smoke pass
                import numpy as _np
                f(_np.zeros((1, 13), dtype=_np.float32))
                return (True, 'pure-python')
            except Exception as e:
                return (False, f'Failed to load ONNX model with pure-Python loader: {e}')

    except Exception as e:
        return (False, str(e))



st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.write("Enter patient data below to predict heart disease risk using our trained neural network.")

st.markdown("---")


def _severity_label_and_color(p):
    """Map probability to a severity label and color."""
    if p < 0.2:
        return "Very low", "#16a34a"  # green
    if p < 0.4:
        return "Low", "#60a5fa"  # blue
    if p < 0.6:
        return "Moderate", "#f59e0b"  # amber
    if p < 0.8:
        return "High", "#f97316"  # orange
    return "Very high", "#dc2626"  # red


def _render_gauge(p, color):
    # Simple CSS gauge bar
    pct = int(round(p * 100))
    gauge_html = f"""
    <div style='width:100%;background:#e6e6e6;border-radius:8px;padding:3px'>
      <div style='width:{pct}%;background:{color};height:28px;border-radius:6px;text-align:right;color:white;padding-right:8px;line-height:28px;font-weight:600'>
        {pct}%
      </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


def _make_result_text(input_dict, proba, pred):
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    label, _ = _severity_label_and_color(proba)
    txt = f"Prediction at {ts}\nProbability: {proba:.4f}\nClass: {pred} ({label})\nInputs: {input_dict}\n"
    return txt


# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0], index=0)[1]
    cp = st.selectbox('Chest Pain Type (0-3)', options=list(range(4)), index=0)
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120)
    chol = st.number_input('Cholesterol (mg/dl)', min_value=50, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1], index=0)[0]
    restecg = st.selectbox('Resting ECG (0-4)', options=list(range(5)), index=0)

with col2:
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1], index=0)[0]
    oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox('ST Segment Slope (0-2)', options=list(range(3)), index=0)
    ca = st.selectbox('Number of Major Vessels (0-3)', options=list(range(4)), index=0)
    thal = st.selectbox('Thal Type (0-4)', options=list(range(5)), index=0)

st.markdown("---")

if st.button('Predict Disease Risk', use_container_width=True):
    # Build input dict with all 13 features
    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': int(fbs),
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }

    try:
        st.info("🔄 Loading model and making prediction...")
        proba, pred = model_module.predict(input_dict)
        
        # Display results (enhanced)
        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2 = st.columns([1, 1])
        with col1:
            label, color = _severity_label_and_color(proba)
            # Big percentage
            st.markdown(f"<div style='font-size:34px;font-weight:700'>Estimated risk: <span style='color:{color}'>{proba*100:.1f}%</span></div>", unsafe_allow_html=True)
            # Gauge
            _render_gauge(proba, color)
            # Severity label and short advice
            st.markdown(f"### Severity: <span style='color:{color}'>{label.title()}</span>", unsafe_allow_html=True)

            # Quick rule-of-thumb notes (simple heuristics)
            notes = []
            if input_dict.get('age', 0) >= 60:
                notes.append('Age ≥ 60 — higher baseline risk')
            if input_dict.get('chol', 0) >= 240:
                notes.append('High cholesterol (≥240 mg/dl) — risk factor')
            if input_dict.get('exang', 0) == 1:
                notes.append('Exercise-induced angina present — concerning')
            if input_dict.get('oldpeak', 0) >= 1.0:
                notes.append('ST depression (oldpeak) ≥ 1.0 — possible ischemia')

            if notes:
                st.markdown("**Clinical flags:**")
                for n in notes:
                    st.write(f"- {n}")
            else:
                st.write("No immediate clinical flags from the provided inputs.")

        with col2:
            st.metric(label="Disease Probability", value=f"{proba:.2%}", delta=None)

            # Download / copy result text
            result_text = _make_result_text(input_dict, proba, pred)
            st.download_button("Download result summary", data=result_text, file_name="prediction.txt", mime="text/plain")

            # Longer interpretation and recommended next steps
            st.markdown("---")
            st.subheader("Interpretation & Next Steps")
            if pred == 1:
                st.warning("⚠️ **Positive prediction**: The model indicates a higher likelihood of heart disease. This is a risk assessment, not a diagnosis. Recommend: consult a cardiologist, obtain ECG/stress test as clinically indicated, and review lipid and BP control.")
            else:
                st.success("✅ **Negative prediction**: Lower likelihood based on the available inputs. Continue routine care and follow-up. If symptoms change, seek medical evaluation.")

            st.markdown("---")
            st.subheader("Input Summary")
            st.json(input_dict)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("This may take a moment on first run as the model is being trained from the UCI dataset.")

st.markdown("---")
st.markdown("**Note**: This model is trained on the UCI Heart Disease dataset. "
           "It should not be used as a substitute for professional medical diagnosis.")
