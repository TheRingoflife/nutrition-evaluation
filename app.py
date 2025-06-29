import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("This app uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy**, based on simplified input features.")

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    return joblib.load("XGBoost_retrained_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler2.pkl")

@st.cache_resource
def load_background_data():
    return np.load("background_data.npy")  # 使用 np.load 加载 .npy 文件

model = load_model()
scaler = load_scaler()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)

# ===== 模型预测 + SHAP 可解释性 =====
if st.sidebar.button("🧮 Predict"):
    # 1. 获取scaler期望的特征名
    expected_columns = list(scaler.feature_names_in_)
    input_dict = {
        "Protein": protein,
        "Sodium": sodium,
        "Energy": energy
    }
    # 2. 检查是否缺失
    missing = [feat for feat in expected_columns if input_dict.get(feat, None) in [None, ""]]
    if missing:
        st.error(f"缺少输入项: {missing}")
        st.stop()
    # 3. 按顺序组装DataFrame
    user_input_for_scaler = pd.DataFrame([[input_dict[feat] for feat in expected_columns]], columns=expected_columns)
    # 4. 标准化
    user_scaled_part = scaler.transform(user_input_for_scaler)
    user_scaled_df = pd.DataFrame(user_scaled_part, columns=expected_columns)

    # 7. 按模型需要的顺序排列
    final_columns = ['Protein', 'Sodium', 'Energy']
    user_scaled_df = user_scaled_df[final_columns]
    # 8. 预测
    prediction = model.predict(user_scaled_df)[0]
    prob = model.predict_proba(user_scaled_df)[0][1]
    # 9. 展示结果
    st.subheader("🔍 Prediction Result")
    label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")
    # 10. SHAP力图
    st.subheader("📊 SHAP Force Plot (Model Explanation)")
    with st.expander("Click to view SHAP force plot"):
        shap_values = explainer(user_scaled_df)
        if isinstance(shap_values, list):  # 如果返回的是列表
            shap_values = shap_values[1]
        if not isinstance(shap_values, shap.Explanation):
            shap_values = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=user_scaled_df.values,
                feature_names=user_scaled_df.columns.tolist()
            )
        force_html = shap.force_plot(
            base_value=shap_values.base_values,
            shap_values=shap_values.values,
            features=shap_values.data,
            feature_names=shap_values.feature_names,
            matplotlib=False
        )
        components.html(shap.getjs() + force_html.html(), height=400)  # 增加高度

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
