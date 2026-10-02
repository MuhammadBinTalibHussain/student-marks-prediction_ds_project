"""
CS 4048 Project I - Student Marks Prediction dashboard (Streamlit)
Run with:  streamlit run app.py
Needs in the same folder: preprocessed_dataset.csv, workflow_pipeline.png
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
RQS = {
    "RQ1 - Midterm I":  {"target": "mid1_pct",  "features": ["assign_avg_early", "quiz_avg_early"]},
    "RQ2 - Midterm II": {"target": "mid2_pct",  "features": ["assign_avg_75", "quiz_avg_75", "mid1_pct"]},
    "RQ3 - Final exam": {"target": "final_pct", "features": ["assign_avg_all", "quiz_avg_all", "mid1_pct", "mid2_pct"]},
}
NICE = {"assign_avg_early": "Average assignment % (first 50%)", "quiz_avg_early": "Average quiz % (first 50%)",
        "assign_avg_75": "Average assignment % (first 75%)", "quiz_avg_75": "Average quiz % (first 75%)",
        "assign_avg_all": "Average assignment % (all)", "quiz_avg_all": "Average quiz % (all)",
        "mid1_pct": "Midterm I %", "mid2_pct": "Midterm II %", "final_pct": "Final exam %"}

st.set_page_config(page_title="Student Marks Prediction", layout="wide")


# ----------------------------------------------------------------------------- helpers (same logic as the notebook)
def make_pipeline(degree=1):
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("regression", LinearRegression()))
    return Pipeline(steps)


def get_scores(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p)}


def bootstrap_mae(model, X, y, n_boot=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n, maes = len(X), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) == 0:
            continue
        m = clone(model).fit(X.iloc[idx], y.iloc[idx])
        maes.append(mean_absolute_error(y.iloc[oob], m.predict(X.iloc[oob])))
    maes = np.array(maes)
    return maes, np.percentile(maes, [2.5, 97.5])


@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_dataset.csv")


@st.cache_resource
def run_rq(rq_name):
    df = load_data()
    cfg = RQS[rq_name]
    target, features = cfg["target"], cfg["features"]
    tr = df[df["split"] == "train"].dropna(subset=[target]).reset_index(drop=True)
    te = df[df["split"] == "test"].dropna(subset=[target]).reset_index(drop=True)
    y_tr, y_te = tr[target], te[target]

    corr = tr[features + [target]].corr()[target].drop(target)
    simple_feature = corr.abs().idxmax()
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    deg_cv = {d: -cross_val_score(make_pipeline(d), tr[features], y_tr, cv=kf,
                                  scoring="neg_mean_absolute_error").mean() for d in (2, 3, 4)}
    best_deg = min(deg_cv, key=deg_cv.get)

    models = {
        "Dummy (mean)": (DummyRegressor(), features),
        f"Simple LR ({simple_feature})": (make_pipeline(1), [simple_feature]),
        "Multiple LR": (make_pipeline(1), features),
        f"Polynomial (deg {best_deg})": (make_pipeline(best_deg), features),
    }
    rows, fitted, boots = [], {}, {}
    for name, (m, feats) in models.items():
        cv = -cross_val_score(m, tr[feats], y_tr, cv=kf, scoring="neg_mean_absolute_error")
        m.fit(tr[feats], y_tr)
        s_tr, s_te = get_scores(y_tr, m.predict(tr[feats])), get_scores(y_te, m.predict(te[feats]))
        boot, ci = bootstrap_mae(m, tr[feats], y_tr)
        rows.append({"Model": name, "CV MAE": round(cv.mean(), 2), "Train MAE": round(s_tr["MAE"], 2),
                     "Test MAE": round(s_te["MAE"], 2), "Test RMSE": round(s_te["RMSE"], 2),
                     "Test R2": round(s_te["R2"], 3), "MAE 95% CI (bootstrap)": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                     "_train": s_tr, "_test": s_te})
        fitted[name], boots[name] = (m, feats), boot
    table = pd.DataFrame(rows)
    best = table.loc[table["CV MAE"].idxmin(), "Model"]      # chosen on TRAIN cross-validation only
    return dict(table=table, fitted=fitted, boots=boots, best=best, te=te, tr=tr, target=target, features=features)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Student Marks Prediction")
page = st.sidebar.radio("Go to", ["Overview & workflow", "EDA", "Model results", "Try a prediction"])
st.sidebar.caption("CS 4048 - Data Science - Project I")

df = load_data()

# ----------------------------------------------------------------------------- pages
if page == "Overview & workflow":
    st.title("How accurately can we predict student marks?")
    st.write("Three research questions: predict **Midterm I** (RQ1), **Midterm II** (RQ2) and the **Final exam** (RQ3) "
             "from the information that is available *before* each exam.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(df)); c2.metric("Courses", df["course"].nunique())
    c3.metric("Train rows", int((df["split"] == "train").sum())); c4.metric("Test rows", int((df["split"] == "test").sum()))
    st.subheader("Workflow / pipeline")
    st.image("workflow_pipeline.png")
    st.subheader("Avoiding data leakage")
    st.markdown("""
- The data is **split first** (80/20, stratified by course).
- Missing-value imputation and scaling are inside a scikit-learn **Pipeline**, so they learn from training rows only.
- Polynomial degree and the best model are chosen with **5-fold CV on the training set**, never on the test set.
- Bootstrapping (500 resamples) uses **training data only**.
- Each RQ only uses assessments that happen **before** the exam (no Midterm II to predict Midterm I).
""")
    st.subheader("Preprocessed dataset")
    st.dataframe(df, width="stretch")

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    col = st.selectbox("Choose a column to explore", [c for c in NICE if c in df.columns], format_func=lambda c: NICE[c])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue"); ax.set_xlabel(NICE[col]); ax.set_title("Distribution")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=df, x="course", y=col, ax=ax, palette="Set2"); ax.set_ylabel(NICE[col]); ax.set_title("By course")
        st.pyplot(fig)
    st.subheader("Correlation heatmap")
    cols = [c for c in NICE if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    st.subheader("Missing values")
    miss = df[cols].isna().sum()
    st.dataframe(miss[miss > 0].rename("missing").to_frame())
    st.subheader("Summary statistics")
    st.dataframe(df[cols].describe().T.round(2))

elif page == "Model results":
    st.title("Model results")
    rq = st.selectbox("Research question", list(RQS))
    with st.spinner("Training models and running 500 bootstrap samples..."):
        res = run_rq(rq)
    st.subheader("Comparison table")
    st.dataframe(res["table"].drop(columns=["_train", "_test"]), width="stretch", hide_index=True)
    st.caption("MAE / RMSE are in percentage points of the exam. Best model = lowest 5-fold CV MAE on training data.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(list(res["boots"].values()), labels=[n.split(" (")[0] for n in res["boots"]], showmeans=True)
        ax.set_title("Bootstrap MAE (500 resamples, train data)"); ax.set_ylabel("MAE")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([n.split(" (")[0] for n in res["table"]["Model"]], res["table"]["Test MAE"],
               color=["#999", "#6baed6", "#3182bd", "#08519c"])
        ax.set_title("Test MAE (lower is better)")
        st.pyplot(fig)

    st.subheader(f"Best model: {res['best']}")
    row = res["table"][res["table"]["Model"] == res["best"]].iloc[0]
    st.table(pd.DataFrame({"Train": row["_train"], "Test": row["_test"]}).round(3))
    gap = row["_train"]["R2"] - row["_test"]["R2"]
    if row["_train"]["R2"] < 0.30 and row["_test"]["R2"] < 0.30:
        st.warning("Low R2 on both train and test: the model is limited by how little the inputs explain (underfitting), not by memorising.")
    elif gap > 0.10:
        st.error("Train R2 is much higher than test R2: overfitting.")
    else:
        st.success("Train and test scores are close: no strong over/underfitting.")

    model, feats = res["fitted"][res["best"]]
    pred = model.predict(res["te"][feats])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(res["te"][res["target"]], pred, alpha=0.7); ax.plot([0, 100], [0, 100], "r--")
    ax.set_xlabel("Actual %"); ax.set_ylabel("Predicted %"); ax.set_title("Test set: predicted vs actual")
    st.pyplot(fig)

else:
    st.title("Try a prediction")
    st.write("Enter a student's marks (in %) and get a predicted exam score from the best model.")
    rq = st.selectbox("What do you want to predict?", list(RQS))
    res = run_rq(rq)
    model, feats = res["fitted"][res["best"]]
    st.caption(f"Using: {res['best']}")
    values = {}
    for f in res["features"]:
        values[f] = st.slider(NICE[f], 0.0, 100.0, float(round(df[f].median(), 1)), 0.5)
    X_new = pd.DataFrame([values])[feats]
    p = float(np.clip(model.predict(X_new)[0], 0, 100))
    mae = float(res["table"][res["table"]["Model"] == res["best"]]["Test MAE"].iloc[0])
    st.metric("Predicted score", f"{p:.1f}%")
    st.info(f"Typical error of this model on unseen students is about +/- {mae:.1f} percentage points, "
            f"so a reasonable range is {max(0, p - mae):.0f}% to {min(100, p + mae):.0f}%.")
