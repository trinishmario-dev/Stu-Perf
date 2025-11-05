# app.py
# -----------------------------------------
# 🧑‍🎓 Student Performance Predictor
# Tech: Streamlit + scikit-learn
# -----------------------------------------

import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data
def get_sample_dataset(rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Create a realistic sample dataset with:
    - attendance_pct (0-100)
    - internal1, internal2 (0-30 each)
    - study_hours_per_week (0-40)
    - prior_gpa (0-10)
    - extracurricular (0/1)
    Target:
    - final_score (0-100)
    """
    rng = np.random.default_rng(random_state)
    attendance = rng.uniform(50, 100, rows).round(1)
    internal1 = rng.integers(10, 30, rows)
    internal2 = rng.integers(10, 30, rows)
    study_hours = rng.uniform(0, 40, rows).round(1)
    prior_gpa = rng.uniform(0, 10, rows).round(2)
    extracurricular = rng.integers(0, 2, rows)

    # Generate target with some noise and nonlinear interactions
    base = (
        0.35 * attendance
        + 1.0 * internal1
        + 1.1 * internal2
        + 0.9 * np.sqrt(np.clip(study_hours, 0, None)) * 5
        + 2.2 * prior_gpa
        + 2.5 * extracurricular
    )
    noise = rng.normal(0, 5, rows)
    final = np.clip(base + noise, 0, 100).round(1)

    df = pd.DataFrame({
        "attendance_pct": attendance,
        "internal1": internal1,
        "internal2": internal2,
        "study_hours_per_week": study_hours,
        "prior_gpa": prior_gpa,
        "extracurricular": extracurricular,
        "final_score": final
    })
    return df

def validate_columns(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found."
    if df.select_dtypes(include=[np.number]).shape[1] < 2:
        return False, "Dataset must contain at least one numeric feature column besides the target."
    return True, ""

def nice_metric_card(label: str, value: float, suffix: str = ""):
    st.metric(label, f"{value:.3f}{suffix}")

def downloadable_bytes(obj, filename="model.pkl"):
    bio = io.BytesIO()
    pickle.dump(obj, bio)
    bio.seek(0)
    return bio

# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Predictor")
st.caption("Train a regression model to predict final scores from attendance, internals, and more.")

with st.sidebar:
    st.header("⚙ Configuration")
    data_source = st.radio("Choose data source", ["Use sample dataset", "Upload CSV"], index=0)

    if data_source == "Use sample dataset":
        n_rows = st.slider("Sample rows", 50, 2000, 400, step=50)
    uploaded = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (numeric features + a target column)", type=["csv"])

    st.divider()
    st.subheader("Model Settings")
    n_estimators = st.slider("n_estimators (trees)", 50, 500, 250, step=25)
    max_depth = st.slider("max_depth", 2, 30, 12, step=1)
    test_size = st.slider("Test size (validation split)", 0.1, 0.4, 0.2, step=0.05)

    st.divider()
    st.subheader("Target Column")
    default_target = "final_score"
    target_col_input = st.text_input("Target column name", value=default_target)

# -----------------------------
# Load Data
# -----------------------------
if data_source == "Use sample dataset":
    df = get_sample_dataset(rows=n_rows)
else:
    if uploaded is None:
        st.warning("Please upload a CSV to continue, or switch to the sample dataset.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

st.subheader("📄 Dataset Preview")
st.write(f"Rows: *{df.shape[0]}, Columns: **{df.shape[1]}*")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# Target & Features Selection
# -----------------------------
valid, msg = validate_columns(df, target_col_input)
if not valid:
    st.error(msg)
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Candidate features = all numeric columns except the target
candidate_features = [c for c in numeric_cols if c != target_col_input]
if not candidate_features:
    st.error("No numeric feature columns available besides the target.")
    st.stop()

with st.expander("🧩 Feature Selection", expanded=True):
    selected_features = st.multiselect(
        "Choose feature columns for training",
        candidate_features,
        default=candidate_features
    )
    if len(selected_features) == 0:
        st.warning("Select at least one feature.")
        st.stop()

# Drop rows with missing values in selected columns
clean_df = df[selected_features + [target_col_input]].dropna()
dropped = len(df) - len(clean_df)
if dropped > 0:
    st.info(f"Dropped {dropped} rows with missing values in selected columns.")

X = clean_df[selected_features]
y = clean_df[target_col_input]

# -----------------------------
# Train/Test Split & Training
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=123
)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=123,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)

st.subheader("📊 Model Performance")
c1, c2, c3 = st.columns(3)
with c1:
    nice_metric_card("R² (Validation)", r2)
with c2:
    nice_metric_card("MAE", mae)
with c3:
    nice_metric_card("RMSE", rmse)

# -----------------------------
# Feature Importance
# -----------------------------
importances = getattr(model, "feature_importances_", None)
if importances is not None:
    st.subheader("🌟 Feature Importance")
    imp_df = pd.DataFrame({"feature": selected_features, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    # Plotly bar chart (Streamlit-native)
    try:
        import plotly.express as px
        fig = px.bar(
            imp_df,
            x="feature",
            y="importance",
            title="Feature Importance",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Importance", bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write(imp_df)

# -----------------------------
# Inference UI (Prediction Form)
# -----------------------------
st.subheader("🧮 Predict a Student's Score")
with st.form("prediction_form"):
    inputs = {}
    for col in selected_features:
        col_min = float(np.nan_to_num(X[col].min(), nan=0.0))
        col_max = float(np.nan_to_num(X[col].max(), nan=100.0))
        col_mean = float(np.nan_to_num(X[col].mean(), nan=(col_min + col_max) / 2))

        # Use number_input with sensible defaults
        inputs[col] = st.number_input(
            f"{col}",
            value=float(np.round(col_mean, 2)),
            help=f"Typical range: {np.round(col_min,2)} – {np.round(col_max,2)}"
        )
    submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([inputs])
        pred = model.predict(row)[0]
        st.success(f"Predicted *{target_col_input}: **{pred:.2f}*")

# -----------------------------
# Downloads: Trained Model & Sample Data
# -----------------------------
st.subheader("📥 Downloads")
colA, colB = st.columns(2)

with colA:
    model_bytes = downloadable_bytes(
        {"model": model, "features": selected_features, "target": target_col_input}
    )
    st.download_button(
        label="Download trained model (pickle)",
        data=model_bytes,
        file_name="student_performance_model.pkl",
        mime="application/octet-stream"
    )

with colB:
    if data_source == "Use sample dataset":
        st.download_button(
            label="Download sample dataset (CSV)",
            data=get_sample_dataset(rows=n_rows).to_csv(index=False).encode("utf-8"),
            file_name="sample_student_performance.csv",
            mime="text/csv"
        )

# -----------------------------
# Help & Tips
# -----------------------------
with st.expander("ℹ Tips & Notes"):
    st.markdown(
        """
- *Upload CSV requirements:* numeric features + a numeric target column (default final_score, configurable in the sidebar).
- You can choose which feature columns to train on under *Feature Selection*.
- Try tuning *n_estimators, **max_depth, and **test_size* to balance bias/variance.
- The downloaded pickle contains the model and the list of expected feature columns.
        """
    )