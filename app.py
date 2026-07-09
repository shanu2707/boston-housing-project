import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# 1. Configured Page Layout
st.set_page_config(page_title="Boston Housing Engine", layout="wide")

st.title("Boston Housing Market Analytics & Predictive Engine")
st.markdown("---")

# 2. Safely Load Cleaned Artifacts
@st.cache_data
def load_data():
    # Reading columns as uppercase to match input standard styles
    data = pd.read_csv('data/cleaned_housing_data.csv')
    data.columns = data.columns.str.upper()
    return data

df = load_data()

with open('models/ridge_housing_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 3. Sidebar Setup - Interactive Pricing Simulator
st.sidebar.header("Real Estate Value Simulator")
st.sidebar.markdown("Adjust the core markers below to run live price predictions.")

# Dynamic Sliders configured explicitly to your dataset parameters
crim = st.sidebar.slider("Crime Rate (CRIM)", float(df['CRIM'].min()), float(df['CRIM'].max()), float(df['CRIM'].mean()))
zn = st.sidebar.slider("Residential Zoning % (ZN)", float(df['ZN'].min()), float(df['ZN'].max()), float(df['ZN'].mean()))
indus = st.sidebar.slider("Non-retail Business % (INDUS)", float(df['INDUS'].min()), float(df['INDUS'].max()), float(df['INDUS'].mean()))
chas = st.sidebar.selectbox("Bounds Charles River? (CHAS)", [0, 1])
nox = st.sidebar.slider("Nitric Oxides Conc. (NOX)", float(df['NOX'].min()), float(df['NOX'].max()), float(df['NOX'].mean()))
rm = st.sidebar.slider("Avg Rooms per Dwelling (RM)", float(df['RM'].min()), float(df['RM'].max()), float(df['RM'].mean()))
age = st.sidebar.slider("Pre-1940 Units % (AGE)", float(df['AGE'].min()), float(df['AGE'].max()), float(df['AGE'].mean()))
dis = st.sidebar.slider("Distance to Employment Hubs (DIS)", float(df['DIS'].min()), float(df['DIS'].max()), float(df['DIS'].mean()))
rad = st.sidebar.slider("Highway Accessibility Index (RAD)", int(df['RAD'].min()), int(df['RAD'].max()), int(df['RAD'].mean()))
tax = st.sidebar.slider("Property Tax Rate (TAX)", float(df['TAX'].min()), float(df['TAX'].max()), float(df['TAX'].mean()))
ptratio = st.sidebar.slider("Pupil-Teacher Ratio (PTRATIO)", float(df['PTRATIO'].min()), float(df['PTRATIO'].max()), float(df['PTRATIO'].mean()))
b = st.sidebar.slider("Bacial Metrics Factor (B)", float(df['B'].min()), float(df['B'].max()), float(df['B'].mean()))
lstat = st.sidebar.slider("% Lower Status Population (LSTAT)", float(df['LSTAT'].min()), float(df['LSTAT'].max()), float(df['LSTAT'].mean()))

# 4. Tab Layout Split
tab1, tab2 = st.tabs(["Market Exploration Dashboard", "Live Valuation Engine"])

with tab1:
    # KPI Summaries Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Valuation (MEDV)", f"${df['MEDV'].mean()*1000:,.0f}")
    col2.metric("Avg Rooms / Dwelling", f"{df['RM'].mean():.1f}")
    col3.metric("Highest Value Track", f"${df['MEDV'].max()*1000:,.0f}")
    col4.metric("Cleaned Samples Profiled", len(df))
    
    st.markdown("### Interactive Trend Explorers")
    left_chart, right_chart = st.columns(2)
    
    with left_chart:
        x_target = st.selectbox("Select Feature for X-Axis:", df.columns[:-1], index=5)
        fig_scatter = px.scatter(df, x=x_target, y="MEDV", trendline="ols",
                                 title=f"Impact of {x_target} on Market Price",
                                 color="LSTAT", color_continuous_scale="Viridis")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with right_chart:
        fig_corr = px.imshow(df.corr(), text_auto=".2f", aspect="auto",
                             title="Feature Correlation Matrix Heatmap", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.markdown("### Real-Time Model Pricing Output")
    
    # Bundle user slider changes into array formats matching training
    input_features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    scaled_inputs = scaler.transform(input_features)
    predicted_val = model.predict(scaled_inputs)[0]
    
    # Beautiful Gauge Display for Predictions
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = float(predicted_val * 1000),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Median Home Value ($)", 'font': {'size': 22}},
        gauge = {
            'axis': {'range': [0, 50000]},
            'bar': {'color': "#2b5c8f"},
            'steps': [
                {'range': [0, 22000], 'color': '#fbc4b6'},
                {'range': [22000, 37000], 'color': '#fdedc4'},
                {'range': [37000, 50000], 'color': '#ccece6'}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)