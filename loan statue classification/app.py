import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 25px;
}
.result-box-low {
    background-color: #dcfce7;
    border: 2px solid #16a34a;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.result-box-high {
    background-color: #fee2e2;
    border: 2px solid #dc2626;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.result-text-low  { color: #16a34a; font-size: 2rem; font-weight: 800; }
.result-text-high { color: #dc2626; font-size: 2rem; font-weight: 800; }
.metric-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.95rem;
}
.metric-label { color: #6b7280; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; }
.metric-value { color: #111827; font-size: 1.1rem; font-weight: 700; }
.age-info {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #1e40af;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL & SCALER
# =========================================================
@st.cache_resource
def load_artifacts():
    errors = []
    model, scaler = None, None
    try:
        model = joblib.load("xgboost_credit_risk.pkl")
    except Exception as e:
        errors.append(f"Model: {e}")
    try:
        scaler = joblib.load("robust_scaler.pkl")
    except Exception as e:
        errors.append(f"Scaler: {e}")
    return model, scaler, errors

model, scaler, load_errors = load_artifacts()


# =========================================================
# HELPERS
# =========================================================
def safe_div(num, den):
    return 0.0 if (den == 0 or den is None) else num / den


def get_age_group(age: int) -> str:
    result = pd.cut(
        [age],
        bins=[18, 25, 35, 50, 65, 84],
        labels=["Young", "Early_Adult", "Mid_Adult", "Senior", "Old"]
    )[0]
    return str(result) if pd.notna(result) else "Young"


AGE_GROUP_MAP = {"Young": 0, "Early_Adult": 1, "Mid_Adult": 2, "Senior": 3, "Old": 4}

# OHE reference categories (drop_first=True → alphabetically first dropped)
# home_ownership: MORTGAGE dropped → keep OTHER, OWN, RENT
HOME_CATS   = ["OTHER", "OWN", "RENT"]
# loan_intent: DEBTCONSOLIDATION dropped → keep rest
INTENT_CATS = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]

GRADE_MAP = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """
    Reproduces the EXACT preprocessing pipeline from the updated notebook:
      1. log1p on person_income & loan_amnt
      2. Feature engineering (NO emp_to_age_ratio, NO person_age in output)
      3. age_group → ordinal (0–4)
      4. loan_grade → ordinal (7–1)
      5. cb_person_default_on_file → binary
      6. OHE on home_ownership & loan_intent (drop_first=True)
    """
    # Raw inputs
    income_raw  = inputs["person_income"]
    loan_raw    = inputs["loan_amnt"]
    emp_length  = inputs["person_emp_length"]
    int_rate    = inputs["loan_int_rate"]
    pct_income  = inputs["loan_percent_income"]
    cb_hist     = inputs["cb_person_cred_hist_length"]
    cb_default  = inputs["cb_person_default_on_file"]
    home        = inputs["person_home_ownership"]
    intent      = inputs["loan_intent"]
    grade       = inputs["loan_grade"]
    age_group   = inputs["age_group"]

    # Step 1 – log transform
    income_log = np.log1p(income_raw)
    loan_log   = np.log1p(loan_raw)

    # Step 2 – engineered features (matches notebook Cell 31–32)
    income_after_loan = income_log - loan_log
    interest_burden   = safe_div(int_rate * loan_log, income_log)
    risk_score        = pct_income + interest_burden + safe_div(1, cb_hist + 1)

    # Step 3 – ordinal age_group
    age_group_enc = AGE_GROUP_MAP[age_group]

    # Step 4 – ordinal loan_grade
    grade_enc = GRADE_MAP[grade]

    # Step 5 – binary cb_person_default_on_file
    default_enc = 1 if cb_default == "Yes" else 0

    # Build base row (person_age NOT included – dropped in notebook Cell 41)
    row = {
        "person_income":             income_log,
        "person_emp_length":         emp_length,
        "loan_grade":                grade_enc,
        "loan_amnt":                 loan_log,
        "loan_int_rate":             int_rate,
        "loan_percent_income":       pct_income,
        "cb_person_default_on_file": default_enc,
        "cb_person_cred_hist_length": cb_hist,
        "age_group":                 age_group_enc,
        "income_after_loan":         income_after_loan,
        "interest_burden":           interest_burden,
        "risk_score":                risk_score,
    }

    # Step 6 – OHE home_ownership (drop MORTGAGE)
    for cat in HOME_CATS:
        row[f"person_home_ownership_{cat}"] = 1 if home == cat else 0

    # Step 6 – OHE loan_intent (drop DEBTCONSOLIDATION)
    for cat in INTENT_CATS:
        row[f"loan_intent_{cat}"] = 1 if intent == cat else 0

    df_row = pd.DataFrame([row])

    # Align column order to what the model was trained on
    expected_cols = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            expected_cols = model.get_booster().feature_names
        except Exception:
            pass

    if expected_cols:
        for col in expected_cols:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row[expected_cols]

    return df_row


def validate(person_income, loan_amnt, person_emp_length, person_age=None):
    errs = []
    if person_income <= 0:
        errs.append("Income must be greater than 0.")
    if loan_amnt <= 0:
        errs.append("Loan amount must be greater than 0.")
    if person_emp_length < 0:
        errs.append("Employment length cannot be negative.")
    if loan_amnt > person_income * 15:
        errs.append("Loan amount seems unrealistically high vs income – please verify.")
    return errs


# =========================================================
# HEADER
# =========================================================
st.markdown('<p class="main-title">🏦 Credit Risk Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Predicts whether a loan applicant is likely to default '
    '— powered by a tuned XGBoost model with RobustScaler.</p>',
    unsafe_allow_html=True
)

if load_errors:
    for err in load_errors:
        if "Scaler" in err:
            st.warning(
                "⚠️ **robust_scaler.pkl not found.**  \n"
                "Add this line to your notebook after fitting the scaler and re-run:  \n"
                "```python\njoblib.dump(scaler, 'robust_scaler.pkl')\n```"
            )
        if "Model" in err:
            st.error(f"❌ Could not load model — {err}")
            st.stop()


# =========================================================
# SIDEBAR — INPUTS
# =========================================================
with st.sidebar:
    st.header("📋 Applicant Information")

    # ── Personal ──────────────────────────────────────────
    st.subheader("👤 Personal Details")

    person_age = st.slider("Age", min_value=18, max_value=84, value=30, step=1)
    age_group  = get_age_group(person_age)

    st.markdown(
        f"""<div class="age-info">
        <b>Auto-detected Age Group:</b> <code>{age_group}</code><br><br>
        18–25 → <b>Young</b> &nbsp;|&nbsp; 25–35 → <b>Early Adult</b><br>
        35–50 → <b>Mid Adult</b> &nbsp;|&nbsp; 50–65 → <b>Senior</b><br>
        65–84 → <b>Old</b>
        </div>""",
        unsafe_allow_html=True
    )

    person_income = st.number_input(
        "Annual Income ($)", min_value=1_000, max_value=2_000_000,
        value=50_000, step=500
    )
    person_emp_length = st.number_input(
        "Employment Length (years)", min_value=0.0, max_value=60.0,
        value=5.0, step=0.5
    )
    person_home_ownership = st.selectbox(
        "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )

    # ── Loan ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Loan Details")

    loan_intent = st.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )
    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"],
        help="A = best creditworthiness, G = worst"
    )
    loan_amnt = st.number_input(
        "Loan Amount ($)", min_value=100, max_value=1_000_000,
        value=10_000, step=100
    )
    loan_int_rate = st.slider(
        "Interest Rate (%)", min_value=0.0, max_value=40.0,
        value=11.0, step=0.1
    )
    loan_percent_income = st.slider(
        "Loan as % of Income (0–1)", min_value=0.0, max_value=1.0,
        value=round(min(loan_amnt / max(person_income, 1), 1.0), 2),
        step=0.01,
        help="Typically loan_amnt / person_income"
    )

    # ── Credit History ────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Credit History")

    cb_person_default_on_file = st.radio(
        "Previous Default on File?", ["No", "Yes"], horizontal=True
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)", min_value=0, max_value=60, value=5, step=1
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")


# =========================================================
# MAIN LAYOUT
# =========================================================
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("📈 Prediction Result")
    result_placeholder = st.container()

with right_col:
    st.subheader("🧮 Engineered Features")
    eng_placeholder = st.container()


# =========================================================
# PREDICTION
# =========================================================
if predict_btn:
    errs = validate(person_income, loan_amnt, person_emp_length)
    if errs:
        for e in errs:
            st.error(f"❌ {e}")
    else:
        inputs = {
            "person_income":             person_income,
            "person_emp_length":         person_emp_length,
            "person_home_ownership":     person_home_ownership,
            "loan_intent":               loan_intent,
            "loan_grade":                loan_grade,
            "loan_amnt":                 loan_amnt,
            "loan_int_rate":             loan_int_rate,
            "loan_percent_income":       loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "age_group":                 age_group,
        }

        # ── Compute engineered features for display (before try) ──
        income_log        = np.log1p(person_income)
        loan_log          = np.log1p(loan_amnt)
        income_after_loan = income_log - loan_log
        interest_burden   = safe_div(loan_int_rate * loan_log, income_log)
        risk_score        = (
            loan_percent_income
            + interest_burden
            + safe_div(1, cb_person_cred_hist_length + 1)
        )

        try:
            feature_row = build_feature_row(inputs)

            # Apply RobustScaler (same as training)
            if scaler is not None:
                feature_arr = scaler.transform(feature_row)
                # Re-wrap as DataFrame so XGBoost keeps feature names
                feature_scaled = pd.DataFrame(
                    feature_arr, columns=feature_row.columns
                )
            else:
                st.warning("⚠️ Scaler not loaded — predicting without scaling (results may be inaccurate).")
                feature_scaled = feature_row

            prediction = model.predict(feature_scaled)[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(feature_scaled)[0][1])

            # ── Result display ────────────────────────────
            with result_placeholder:
                if prediction == 1:
                    st.markdown(
                        """<div class="result-box-high">
                        <div class="result-text-high">⚠️ HIGH RISK</div>
                        <p style="margin:8px 0 0;">This applicant is likely to <b>default</b> on the loan.</p>
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """<div class="result-box-low">
                        <div class="result-text-low">✅ LOW RISK</div>
                        <p style="margin:8px 0 0;">This applicant is likely to <b>repay</b> the loan.</p>
                        </div>""",
                        unsafe_allow_html=True
                    )

                if proba is not None:
                    st.markdown("&nbsp;")
                    st.metric("Probability of Default", f"{proba * 100:.1f}%")
                    st.progress(float(np.clip(proba, 0.0, 1.0)))

                st.success("✅ Prediction completed successfully.")

            # ── Engineered features display ───────────────
            with eng_placeholder:
                def metric_card(label, value):
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                metric_card("📉 Income After Loan (log space)", f"{income_after_loan:.4f}")
                metric_card("💸 Interest Burden",               f"{interest_burden:.4f}")
                metric_card("⚠️ Composite Risk Score",          f"{risk_score:.4f}")
                metric_card("🔖 Age Group (ordinal)",
                            f"{age_group}  →  {AGE_GROUP_MAP[age_group]}")
                metric_card("📊 Loan Grade (ordinal)",
                            f"{loan_grade}  →  {GRADE_MAP[loan_grade]}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

else:
    with result_placeholder:
        st.info("👈 Fill in the applicant details in the sidebar, then click **Predict Risk**.")
    with eng_placeholder:
        st.info("Engineered features will appear here after prediction.")

st.markdown("---")
st.caption(
    "Model: Tuned XGBoost · Scaler: RobustScaler · "
    "Built with Streamlit · For educational / portfolio purposes only."
)
