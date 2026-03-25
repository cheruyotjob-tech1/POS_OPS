import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import textwrap
from datetime import timedelta
import warnings

# === 1. CONFIGURATION & STYLING ===
st.set_page_config(page_title="Daily Deck Dashboard", layout="wide")
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# === 2. DATA LOADING & DERIVED COLUMNS (AT THE TOP) ===
@st.cache_data
def load_and_process_data(file_path):
    # Load data
    df = pd.read_parquet(file_path)
    
    # A. Date Conversions
    df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
    df['ZED_DATE'] = pd.to_datetime(df['ZED_DATE'], errors='coerce')
    
    # B. Numeric Cleaning
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # C. Derived Column: GROSS_SALES
    df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
    
    # D. Derived Column: CUST_CODE (Unique Receipt ID)
    for col in ['STORE_CODE', 'TILL', 'SESSION', 'RCT']:
        df[col] = df[col].astype(str).fillna('').str.strip()
    df['CUST_CODE'] = df['STORE_CODE'] + '-' + df['TILL'] + '-' + df['SESSION'] + '-' + df['RCT']
    
    # E. Derived Column: Till_Code
    df['Till_Code'] = df['TILL'] + '-' + df['STORE_CODE']
    
    # F. Derived Column: Shift Buckets
    df['Shift_Bucket'] = np.where(
        df['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day'
    )
    
    return df

# Initialize Data
DATA_PATH = 'DAILY_POS_TRN_ITEMS_2026-03-22.parquet'
try:
    df = load_and_process_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading file: {e}. Please ensure the parquet file is in the same directory.")
    st.stop()

# === 3. STREAMLIT UI ===
st.title("📊 POS Daily Performance Deck")

tabs = st.tabs(["Sales Overview", "Shift Analysis", "Channel Share", "Operations"])

# --- TAB 1: SALES OVERVIEW ---
with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Global Sales Overview (L1)
        global_sales = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        global_sales['NET_SALES_M'] = global_sales['NET_SALES'] / 1_000_000
        global_sales['PCT'] = (global_sales['NET_SALES'] / global_sales['NET_SALES'].sum()) * 100
        
        fig1 = go.Figure(data=[go.Pie(
            labels=global_sales['SALES_CHANNEL_L1'],
            values=global_sales['NET_SALES_M'],
            hole=0.6,
            textinfo='percent'
        )])
        fig1.update_layout(title="Sales Channel Type (L1)")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Global Net Sales Distribution (L2)
        channel2_sales = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
        fig2 = go.Figure(data=[go.Pie(
            labels=channel2_sales['SALES_CHANNEL_L2'],
            values=channel2_sales['NET_SALES'],
            hole=0.6
        )])
        fig2.update_layout(title="Sales Mode Distribution (L2)")
        st.plotly_chart(fig2, use_container_width=True)

    # Store Sales Summary Table
    st.subheader("Store-wise Sales Summary")
    ss_agg = df.groupby('STORE_NAME', as_index=False).agg({
        'NET_SALES': 'sum',
        'GROSS_SALES': 'sum',
        'CUST_CODE': 'nunique'
    }).rename(columns={'CUST_CODE': 'Customer_Numbers'})
    
    ss_agg['Gross_BV'] = ss_agg['GROSS_SALES'] / ss_agg['Customer_Numbers']
    ss_agg['% Contribution'] = (ss_agg['NET_SALES'] / ss_agg['NET_SALES'].sum()) * 100
    ss_agg = ss_agg.sort_values('NET_SALES', ascending=False).reset_index(drop=True)
    
    st.dataframe(ss_agg.style.format({
        'NET_SALES': '{:,.0f}',
        'GROSS_SALES': '{:,.0f}',
        'Customer_Numbers': '{:,.0f}',
        'Gross_BV': '{:,.2f}',
        '% Contribution': '{:,.2f}%'
    }), use_container_width=True)

# --- TAB 2: SHIFT ANALYSIS ---
with tabs[1]:
    # Night vs Day Ratio
    stores_with_night = df[df['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)].copy()
    
    ratio_df = df_nd.groupby(['STORE_NAME', 'Shift_Bucket'], as_index=False)['NET_SALES'].sum()
    store_totals = ratio_df.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    ratio_df['PCT'] = 100 * ratio_df['NET_SALES'] / store_totals
    pivot_df = ratio_df.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0).sort_values('Night', ascending=False)

    fig_shift = px.bar(pivot_df, orientation='h', title="Night vs Day Sales Ratio per Store",
                       color_discrete_map={'Night': '#d62728', 'Day': '#1f77b4'})
    st.plotly_chart(fig_shift, use_container_width=True)

# --- TAB 3: CHANNEL SHARE ---
with tabs[2]:
    # 2nd Highest Channel Share
    st.subheader("2nd-Highest Channel Share Analysis")
    store_chan = df.groupby(["STORE_NAME", "SALES_CHANNEL_L1"], as_index=False)["NET_SALES"].sum()
    store_tot = store_chan.groupby("STORE_NAME")["NET_SALES"].transform("sum")
    store_chan["PCT"] = 100 * store_chan["NET_SALES"] / store_tot
    store_chan = store_chan.sort_values(["STORE_NAME", "PCT"], ascending=[True, False])
    store_chan["RANK"] = store_chan.groupby("STORE_NAME").cumcount() + 1
    
    second_tbl = store_chan[store_chan["RANK"] == 2][["STORE_NAME", "SALES_CHANNEL_L1", "PCT"]]
    top_30 = second_tbl.sort_values("PCT", ascending=False).head(30)
    
    fig_lollipop = px.scatter(top_30, x="PCT", y="STORE_NAME", text="SALES_CHANNEL_L1",
                             title="Top 30 Stores by 2nd-Highest Channel Share")
    fig_lollipop.update_traces(marker=dict(size=12))
    st.plotly_chart(fig_lollipop, use_container_width=True)

# --- TAB 4: OPERATIONS ---
with tabs[3]:
    st.subheader("Customer Traffic Heatmap (30-min intervals)")
    
    # Heatmap Processing
    df['TIME_INTERVAL'] = df['TRN_DATE'].dt.floor('30min')
    df['TIME_ONLY'] = df['TIME_INTERVAL'].dt.time
    
    traffic = df.groupby(['STORE_NAME', 'TIME_ONLY'])['CUST_CODE'].nunique().reset_index()
    hm_data = traffic.pivot(index='STORE_NAME', columns='TIME_ONLY', values='CUST_CODE').fillna(0)
    
    fig_hm = px.imshow(hm_data, text_auto=True, aspect="auto", color_continuous_scale='Viridis',
                      labels=dict(x="Time", y="Store", color="Receipts"))
    st.plotly_chart(fig_hm, use_container_width=True)
