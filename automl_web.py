import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

st.set_page_config(
    page_title="AutoML Engine",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': None,'Report a bug': None,'About': None}
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1d391kg { max-width: 900px; margin: auto; }
    .subheader { font-size: 2em !important; font-weight: 600; color: #333; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AutoML Engine")
st.markdown('<p class="subheader">Upload data. Choose models. Pick the best.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        if len(df) < 5:
            st.warning("Dataset is very small. Some models may not train properly.")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

if df is not None:
    target_column = st.selectbox("Select target column", df.columns)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    unique_count = len(y.unique())
    if y.dtype == "object" or y.dtype.name == "category":
        task = "Classification"
    elif np.issubdtype(y.dtype, np.integer) and unique_count <= 20:
        task = "Classification"
    else:
        task = "Regression"

    st.write(f"Detected Task Type: {task}")
    st.write("Select models to train")
    model_options = []

    if task == "Classification":
        if st.checkbox("Logistic Regression"):
            model_options.append("LogisticRegression")
        if st.checkbox("Random Forest"):
            model_options.append("RandomForest")
        if st.checkbox("Gradient Boosting"):
            model_options.append("GradientBoostingClassifier")
        if st.checkbox("Support Vector Classifier"):
            model_options.append("SVC")
        if st.checkbox("Ridge Classifier"):
            model_options.append("RidgeClassifier")
        if st.checkbox("K-Nearest Neighbors"):
            model_options.append("KNNClassifier")
    else:
        if st.checkbox("Linear Regression"):
            model_options.append("LinearRegression")
        if st.checkbox("Random Forest Regressor"):
            model_options.append("RandomForestRegressor")
        if st.checkbox("Gradient Boosting Regressor"):
            model_options.append("GradientBoostingRegressor")
        if st.checkbox("Support Vector Regressor"):
            model_options.append("SVR")
        if st.checkbox("Lasso Regression"):
            model_options.append("Lasso")
        if st.checkbox("K-Nearest Neighbors Regressor"):
            model_options.append("KNNRegressor")

    if "results" not in st.session_state:
        st.session_state.results = {}
    if "pipelines" not in st.session_state:
        st.session_state.pipelines = {}
    if "fig" not in st.session_state:
        st.session_state.fig = None

    if st.button("Run AutoML") and len(model_options) > 0:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        models = {}
        if "LogisticRegression" in model_options:
            models["LogisticRegression"] = LogisticRegression(max_iter=1000)
        if "RandomForest" in model_options:
            models["RandomForest"] = RandomForestClassifier()
        if "GradientBoostingClassifier" in model_options:
            models["GradientBoostingClassifier"] = GradientBoostingClassifier()
        if "SVC" in model_options:
            models["SVC"] = SVC(probability=True)
        if "RidgeClassifier" in model_options:
            models["RidgeClassifier"] = RidgeClassifier()
        if "KNNClassifier" in model_options:
            models["KNNClassifier"] = KNeighborsClassifier()
        if "LinearRegression" in model_options:
            models["LinearRegression"] = LinearRegression()
        if "RandomForestRegressor" in model_options:
            models["RandomForestRegressor"] = RandomForestRegressor()
        if "GradientBoostingRegressor" in model_options:
            models["GradientBoostingRegressor"] = GradientBoostingRegressor()
        if "SVR" in model_options:
            models["SVR"] = SVR()
        if "Lasso" in model_options:
            models["Lasso"] = Lasso()
        if "KNNRegressor" in model_options:
            models["KNNRegressor"] = KNeighborsRegressor()

        test_size = 0.2 if len(X) > 1 else 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        results, pipelines = {}, {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            cv = min(3, len(y_train))
            if cv < 2:
                st.warning(f"Not enough samples for cross-validation with {name}. Using training score instead.")
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_train, y_train)
            else:
                score = cross_val_score(pipeline, X_train, y_train, cv=cv).mean()
            results[name] = score
            pipelines[name] = pipeline

        st.session_state.results = results
        st.session_state.pipelines = pipelines

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results.keys(), results.values(), color='skyblue')
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison (Cross-Validation Mean Score)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.session_state.fig = fig

    if st.session_state.results:
        st.write("### Model Comparison")
        st.pyplot(st.session_state.fig)

    if st.session_state.pipelines:
        final_model_name = st.selectbox("Select final model to train fully", list(st.session_state.pipelines.keys()), key="final_model_select")
    else:
        final_model_name = None

    if final_model_name:
        test_size = 0.2 if len(X) > 1 else 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        final_model = st.session_state.pipelines[final_model_name]
        final_model.fit(X_train, y_train)

        if len(X_test) > 0:
            test_score = final_model.score(X_test, y_test)
            st.success(f"Test Set Score for {final_model_name}: {test_score:.4f}")
        else:
            st.info("Dataset too small to split test set. Model trained on all data.")

        st.divider()
        st.subheader("Use the Model for Predictions")
        st.write("Upload a new CSV file to generate predictions using the trained model.")
        predict_file = st.file_uploader("Upload CSV for Prediction", type=["csv"], key="predict_file")
        if predict_file:
            new_data = pd.read_csv(predict_file)
            st.write("Preview of input data:", new_data.head())
            try:
                preds = final_model.predict(new_data)
                pred_df = pd.DataFrame(preds, columns=["Prediction"])
                st.write("Predictions")
                st.dataframe(pred_df)
                st.download_button("Download Predictions as CSV", data=pred_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.divider()
        st.subheader("Export Trained Model")
        st.write("You can export the trained model to use it later in another Python environment.")
        joblib.dump(final_model, "final_model.pkl")
        with open("final_model.pkl", "rb") as f:
            st.download_button(label="Export Model (.pkl)", data=f, file_name="final_model.pkl", mime="application/octet-stream")
    else:
        st.info("Run AutoML and select a model to train fully.")
