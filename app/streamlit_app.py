
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Intelligent Inventory Optimization",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Intelligent Inventory Prediction & Cost Optimization")
st.caption("Explainable AI framework for demand prediction and supply-chain decisions")

@st.cache_data
def load_csv(file_bytes, filename):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def get_data():
    files = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []

    if files:
        selected = st.sidebar.selectbox(
            "Select dataset",
            files,
            format_func=lambda p: p.name,
        )
        try:
            return pd.read_csv(selected), selected.name
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")

    uploaded = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded:
        try:
            return load_csv(uploaded.getvalue(), uploaded.name), uploaded.name
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")

    return None, None

df, source = get_data()

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Explorer",
        "Demand Prediction",
        "Cost Prediction",
        "Explainability",
        "Inventory Optimization",
        "What-if Analysis",
        "Reports",
    ],
)

if page == "Project Overview":
    st.subheader("Project Overview")
    st.write(
        "This dashboard explores inventory-related data, predicts a selected "
        "numeric target, examines influential features, and demonstrates "
        "inventory decision scenarios."
    )

    st.markdown("""
    **Research workflow**
    1. Supply-chain data collection
    2. Data cleaning and validation
    3. Demand prediction
    4. Supply-chain cost prediction
    5. Explainability and what-if analysis
    6. Inventory decision support
    """)

    if df is not None:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        st.success(f"Dataset loaded: {source}")
    else:
        st.info("Choose a CSV dataset from the sidebar to begin.")

if df is not None:
    if page == "Dataset Explorer":
        st.subheader("Dataset Explorer")
        a, b, c = st.columns(3)
        a.metric("Records", f"{len(df):,}")
        b.metric("Features", len(df.columns))
        c.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        st.write("Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.write("Data types and missing values")
        profile = pd.DataFrame({
            "Data type": df.dtypes.astype(str),
            "Missing": df.isna().sum(),
            "Missing (%)": (df.isna().mean() * 100).round(2),
            "Unique values": df.nunique(),
        })
        st.dataframe(profile, use_container_width=True)

        st.download_button(
            "Download dataset",
            df.to_csv(index=False).encode("utf-8"),
            "dataset.csv",
            "text/csv",
        )

    elif page in ["Demand Prediction", "Cost Prediction"]:
        st.subheader(page)

        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric) < 2:
            st.warning("The dataset needs at least two numeric columns.")
        else:
            default_target = 0
            target = st.selectbox(
                "Choose prediction target",
                numeric,
                index=default_target,
                key=f"target_{page}",
            )
            features = st.multiselect(
                "Choose input features",
                [col for col in numeric if col != target],
                default=[col for col in numeric if col != target][:5],
                key=f"features_{page}",
            )

            if st.button("Train and evaluate model", key=f"train_{page}"):
                if not features:
                    st.error("Select at least one input feature.")
                else:
                    work = df[features + [target]].replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()

                    if len(work) < 20:
                        st.error("At least 20 complete rows are recommended.")
                    else:
                        X = work[features]
                        y = work[target]

                        if y.nunique() < 2:
                            st.error("The selected target has no useful variation.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )

                            model = RandomForestRegressor(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1,
                            )
                            with st.spinner("Training model..."):
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)

                            st.session_state[f"model_{page}"] = model
                            st.session_state[f"features_{page}_saved"] = features
                            st.session_state[f"target_{page}_saved"] = target
                            st.session_state[f"predictions_{page}"] = (
                                y_test.to_numpy(), predictions
                            )

            saved = st.session_state.get(f"predictions_{page}")
            if saved is not None:
                actual, predicted = saved
                st.markdown("### Evaluation results")
                a, b, c = st.columns(3)
                a.metric("MAE", f"{mean_absolute_error(actual, predicted):.4f}")
                b.metric(
                    "RMSE",
                    f"{np.sqrt(mean_squared_error(actual, predicted)):.4f}",
                )
                score = r2_score(actual, predicted) if len(actual) > 1 else 0
                c.metric("R²", f"{score:.4f}")

                chart = pd.DataFrame({
                    "Actual": actual,
                    "Predicted": predicted,
                })
                st.plotly_chart(
                    px.scatter(
                        chart,
                        x="Actual",
                        y="Predicted",
                        title="Actual vs predicted values",
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    "Download evaluation predictions",
                    chart.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                    key=f"download_{page}",
                )

    elif page == "Explainability":
        st.subheader("Feature Importance")

        model = st.session_state.get("model_Demand Prediction")
        features = st.session_state.get(
            "features_Demand Prediction_saved"
        )

        if model is None:
            model = st.session_state.get("model_Cost Prediction")
            features = st.session_state.get(
                "features_Cost Prediction_saved"
            )

        if model is None or not features:
            st.info("Train a demand or cost prediction model first.")
        else:
            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)

            st.plotly_chart(
                px.bar(
                    importance,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Random Forest feature importance",
                ),
                use_container_width=True,
            )
            st.dataframe(importance, use_container_width=True)
            st.caption(
                "This is model-based feature importance, not SHAP. "
                "It shows predictive influence, not causation."
            )

    elif page == "Inventory Optimization":
        st.subheader("Inventory Decision Support")
        st.warning(
            "Demonstration only: use real stock, lead-time, and cost inputs "
            "before treating recommendations as operational decisions."
        )

        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            stock_col = st.selectbox("Stock or quantity column", numeric)
            demand_col = st.selectbox(
                "Demand or sales column",
                numeric,
                index=min(1, len(numeric) - 1),
            )
            lead_days = st.number_input(
                "Assumed replenishment lead time (days)",
                min_value=1,
                value=7,
            )
            safety_days = st.number_input(
                "Safety-stock coverage (days)",
                min_value=0,
                value=3,
            )

            if st.button("Calculate indicative reorder recommendations"):
                stock = pd.to_numeric(df[stock_col], errors="coerce")
                demand = pd.to_numeric(df[demand_col], errors="coerce")
                average_demand = demand.mean()

                if pd.isna(average_demand) or average_demand < 0:
                    st.error("Demand data must have a valid non-negative mean.")
                else:
                    reorder_point = average_demand * (lead_days + safety_days)
                    result = pd.DataFrame({
                        "Observed quantity": stock,
                        "Observed demand": demand,
                        "Indicative reorder point": reorder_point,
                    })
                    result["Indicative reorder quantity"] = (
                        reorder_point - stock
                    ).clip(lower=0)

                    st.metric("Average demand in dataset", f"{average_demand:.2f}")
                    st.metric("Indicative reorder point", f"{reorder_point:.2f}")
                    st.dataframe(result.head(100), use_container_width=True)
                    st.download_button(
                        "Download recommendations",
                        result.to_csv(index=False).encode("utf-8"),
                        "inventory_recommendations.csv",
                        "text/csv",
                    )
        else:
            st.warning("No numeric columns are available.")

    elif page == "What-if Analysis":
        st.subheader("What-if Demand Scenario")
        numeric = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric:
            st.warning("No numeric columns are available.")
        else:
            demand_col = st.selectbox("Demand or sales column", numeric)
            change = st.slider(
                "Assumed demand change (%)",
                min_value=-50,
                max_value=100,
                value=10,
                step=5,
            )

            demand = pd.to_numeric(df[demand_col], errors="coerce").dropna()
            baseline = demand.mean()
            scenario = baseline * (1 + change / 100)

            a, b, c = st.columns(3)
            a.metric("Baseline average", f"{baseline:.2f}")
            b.metric("Scenario average", f"{scenario:.2f}")
            c.metric("Change", f"{change}%")

            st.caption(
                "This is a simple arithmetic scenario, not a validated "
                "forecast of how demand will actually change."
            )

    elif page == "Reports":
        st.subheader("Download Reports")
        st.write("Export the currently loaded dataset for further analysis.")
        st.download_button(
            "Download dataset as CSV",
            df.to_csv(index=False).encode("utf-8"),
            "inventory_data_export.csv",
            "text/csv",
        )

else:
    if page != "Project Overview":
        st.info("Select a CSV dataset in the sidebar or upload one to continue.")

st.sidebar.divider()
st.sidebar.caption("Prototype dashboard | Validate models before operational use")
