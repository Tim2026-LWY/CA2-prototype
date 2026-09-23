# This file must work standing ALONE on Streamlit Cloud (no Colab, no Drive).
# Upload Lab04_hk_car_price.csv in the SAME GitHub folder as this file.

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CHANGE THIS LINE. Use your own column names. Do not leave the words PASTE / HERE.
# Do not add Displacement_cc. Empty engine cc will crash training.
FEATURES = ["Manufacture_Year", "Mileage_km", "Displacement_cc"]        # Example: ["Manufacture_Year", "Mileage_km"]
RANDOM_STATE = 42   # public demo — does not need to match your Student ID

@st.cache_data
def load_and_train():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # 读取数据集
    df = pd.read_csv("你的数据文件名.csv")

    # 【重点1】过滤里程=0（防止除以0产生Inf，二手车项目高频坑）
    df = df[df["mileage"] > 0]

    # 【重点2】把无穷大、负无穷转为NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 【重点3】删除所有包含空值的行
    df = df.dropna()

    # 拆分特征X 和目标变量 y（价格）
    X = df.drop("price", axis=1)
    y = df["price"]
    model_columns = list(X.columns)

    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    return df, model, model_columns


df, model, model_columns = load_and_train()

st.title("HK Used Car Price Estimator — LauWingYip_250039758_CA2 Prototype")                  #<- Change your name here!!!=======================================
st.write("Predicts **resale price (HKD)** from real Hong Kong Motor City transactions. This is a quote ballpark — not an official valuation form.")

# Sliders and brand menus are built from FEATURES. Do not delete this loop.
inputs = {}
for col in FEATURES:
    if not pd.api.types.is_numeric_dtype(df[col]):
        inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique().tolist()))
    else:
        inputs[col] = st.slider(
            col, float(df[col].min()), float(df[col].max()), float(df[col].mean())
        )

if st.button("Estimate Price"):
    row = pd.DataFrame([inputs])
    text_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if text_cols:
        row = pd.get_dummies(row, columns=text_cols)
    row = row.reindex(columns=model_columns, fill_value=0)
    price = model.predict(row)[0]
    st.success(f"Estimated price: HK${price:,.0f}")
