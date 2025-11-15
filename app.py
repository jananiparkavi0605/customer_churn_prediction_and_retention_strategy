import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# ------------------- Read Uploaded File Function -------------------
def read_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    fname = uploaded_file.name.lower()

    if fname.endswith((".xls", ".xlsx")):
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            raise

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        pass

    for enc in ["utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16"]:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc, engine="python")
        except Exception:
            continue

    text = raw_bytes.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), engine="python", sep=None)

# ------------------- Upload Dataset -------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read file: {e}")
        st.stop()

    df.dropna(axis=1, how="all", inplace=True)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # ------------------- Correlation Heatmap -------------------
    st.subheader("📈 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("No numeric columns to display correlation heatmap.")

    # ------------------- Detect Churn Column -------------------
    churn_col = None
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ["churn", "exit", "exited", "is_churn", "left"]):
            churn_col = col
            break

    if churn_col:
        st.success(f"✅ Detected churn column: **{churn_col}**")
    else:
        churn_col = st.selectbox("Select Target Column", df.columns)

    if churn_col:
        X = df.drop(columns=[churn_col])
        y = df[churn_col]

        # Fill missing values
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna("missing")

        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Encode target
        y_le = None
        if y.dtype == "object" or y.dtype.name == "category":
            y_le = LabelEncoder()
            y = y_le.fit_transform(y.astype(str))
        if y.dtype == "bool":
            y = y.astype(int)

        try:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale numeric features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # XGBoost with GridSearch
            xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(xgb, param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)

            model = grid.best_estimator_
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

        # ------------------- Metrics -------------------
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.2%}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # ------------------- Confusion Matrix -------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not Churn", "Churn"],
                    yticklabels=["Not Churn", "Churn"],
                    ax=ax_cm)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        st.pyplot(fig_cm)

        # ------------------- ROC Curve -------------------
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

        # ------------------- Feature Importance -------------------
        try:
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
            st.subheader("Top Feature Importances")
            st.bar_chart(feat_imp)
        except Exception:
            pass

        # ------------------- Prediction UI -------------------
        st.subheader("🔮 Try Prediction with Custom Input")
        input_data = {}
        for col in X.columns:
            if col in label_encoders:
                options = list(label_encoders[col].classes_)
                input_data[col] = st.selectbox(f"{col}", options, key=f"inp_{col}")
            else:
                val = float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                input_data[col] = st.number_input(f"{col}", value=val, key=f"num_{col}")

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([input_data])
            for col, le in label_encoders.items():
                try:
                    input_df[col] = le.transform(input_df[col])
                except Exception:
                    input_df[col] = 0
            input_df = scaler.transform(input_df)
            pred = model.predict(input_df)[0]
            if y_le is not None:
                pred_display = y_le.inverse_transform([pred])[0]
            else:
                pred_display = "Churn" if int(pred) == 1 else "Not Churn"
            st.success(f"Prediction: **{pred_display}**")

        # ------------------- Download Full Report -------------------
        st.subheader("📥 Download Full Dashboard Report")

        def create_pdf_report():
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
            styles = getSampleStyleSheet()
            content = []

            content.append(Paragraph("Customer Churn Prediction Report", styles['Title']))
            content.append(Spacer(1, 12))
            content.append(Paragraph(f"Model Accuracy: {accuracy:.2%}", styles['Normal']))
            content.append(Spacer(1, 12))
            content.append(Paragraph("Classification Report:", styles['Heading2']))
            content.append(Paragraph(f"<pre>{classification_report(y_test, y_pred)}</pre>", styles['Normal']))

            # Save images
            fig_cm.savefig("cm_temp.png")
            fig_roc.savefig("roc_temp.png")
            fig_corr.savefig("corr_temp.png")

            content.append(Spacer(1, 12))
            content.append(Paragraph("Confusion Matrix:", styles['Heading2']))
            content.append(Image("cm_temp.png", width=400, height=300))
            content.append(Spacer(1, 12))
            content.append(Paragraph("ROC Curve:", styles['Heading2']))
            content.append(Image("roc_temp.png", width=400, height=300))
            content.append(Spacer(1, 12))
            content.append(Paragraph("Correlation Heatmap:", styles['Heading2']))
            content.append(Image("corr_temp.png", width=400, height=300))

            doc.build(content)
            return temp_file.name

        if st.button("Generate & Download Report"):
            pdf_path = create_pdf_report()
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download Dashboard Report (PDF)",
                    data=f,
                    file_name="Customer_Churn_Report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("⬆️ Upload a CSV or Excel file to get started.")
