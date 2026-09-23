# This file must work standing ALONE on Streamlit Cloud (no Colab, no Drive).
# Upload Lab04_hk_car_price.csv in the SAME GitHub folder as this file.
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# CHANGE THIS LINE. Use your own column names.
FEATURES = ["Manufacture_Year", "Mileage_km", "Displacement_cc"]
RANDOM_STATE = 42   # public demo — does not need to match your Student ID

@st.cache_data
def load_and_train():
    df = pd.read_csv("Lab04_hk_car_price.csv")
    X_all = df[FEATURES].copy()
    y_all = df["Price_HKD"]
    
    # Fill empty numeric cells with column mean to avoid NaN crash
    X_all = X_all.fillna(X_all.mean())

    # Brand is text. Split it into number columns before fit().
    text_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(X_all[c])]
    if text_cols:
        X_all = pd.get_dummies(X_all, columns=text_cols, drop_first=True)
        X_all = X_all.astype(float)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return df, model, list(model.feature_names_in_)

df, model, model_columns = load_and_train()

# ========= IMPROVED MODEL METRICS =========
# Define the list of predictor columns: add Displacement_cc (engine size) to original features
FEATURES = ["Manufacture_Year", "Mileage_km", "Displacement_cc"]
# Select all feature columns from dataframe df
X_all = df[FEATURES]
# Select the target column we want to predict: used car price in HKD
y_all = df["Price_HKD"]
# Handle missing values in feature columns to prevent model fit crash
# Fill empty cells with the average value of that column
X_all = X_all.fillna(X_all.mean())
# Split dataset into training and test sets, keep fixed TEST_SIZE and RANDOM_STATE
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
)
# Create linear regression model object
model = LinearRegression()
# Train (fit) the model using the training data
model.fit(X_train, y_train)
# Predict prices for training set and calculate Train MAE
y_pred_train = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
# Predict prices for unseen test set and calculate Test MAE and Test R²
y_pred_test = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
# Required print statements exactly as specified
print("Features used:", FEATURES)
print(f"Improved Train MAE: HK${train_mae:,.0f}")
print(f"Improved Test MAE: HK${test_mae:,.0f}")
print(f"Improved Test R²: {test_r2:.2f}")
print("\nIf Test MAE is still almost equal to the starter, you have not improved yet.")


st.title("HK Used Car Price Estimator — LauWingYip_250039758_CA2 Prototype")
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

