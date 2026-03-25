# -*- coding: utf-8 -*-
"""
Daily Deck Dashboard - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Daily Deck Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Daily Deck Dashboard - 2026-03-22")
st.markdown("### POS Transaction Analysis")

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    df = pd.read_parquet('DAILY_POS_TRN_ITEMS_2026-03-22.parquet')
    return df

df = load_data()

# ====================== BASIC CLEANING ======================
with st.spinner("Cleaning and preparing data..."):
    # Date conversion
    df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
    df['ZED_DATE'] = pd.to_datetime(df['ZED_DATE'], errors='coerce')

    # Numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Derived columns
    if 'GROSS_SALES' not in df.columns:
        df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']

    # Unique Receipt Code
    for col in ['STORE_CODE', 'TILL', 'SESSION', 'RCT']:
        df[col] = df[col].astype(str).fillna('').str.strip()
    
    df['CUST_CODE'] = (
        df['STORE_CODE'] + '-' +
        df['TILL'] + '-' +
        df['SESSION'] + '-' +
        df['RCT']
    )

    df['Till_Code'] = df['TILL'] + '-' + df['STORE_CODE']
    df['CASHIER-COUNT'] = df['CASHIER'].astype(str).str.strip() + '-' + df['STORE_NAME'].astype(str).str.strip()

st.success(f"✅ Data loaded successfully! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ====================== SIDEBAR FILTERS ======================
st.sidebar.header("🔍 Filters")

selected_stores = st.sidebar.multiselect(
    "Select Stores",
    options=sorted(df['STORE_NAME'].unique()),
    default=sorted(df['STORE_NAME'].unique())[:5] if len(df['STORE_NAME'].unique()) > 5 else None
)

date_range = st.sidebar.date_input(
    "Transaction Date Range",
    value=(df['TRN_DATE'].min().date(), df['TRN_DATE'].max().date())
)

# Apply filters
mask = (df['TRN_DATE'].dt.date >= date_range[0]) & (df['TRN_DATE'].dt.date <= date_range[1])
if selected_stores:
    mask = mask & df['STORE_NAME'].isin(selected_stores)

filtered_df = df[mask].copy()

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sales Overview", 
    "👥 Customer Traffic & Tills", 
    "🛒 Baskets & Products", 
    "💰 Loyalty & Refunds", 
    "📋 Raw Summary Tables"
])

# ====================== TAB 1: SALES OVERVIEW ======================
with tab1:
    st.header("Sales Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Global Sales by Channel L1
        global_sales = filtered_df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
        global_sales['NET_SALES_M'] = global_sales['NET_SALES'] / 1_000_000
        global_sales['PCT'] = global_sales['NET_SALES'] / global_sales['NET_SALES'].sum() * 100

        fig1 = go.Figure(data=[go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}%)" for _, row in global_sales.iterrows()],
            values=global_sales['NET_SALES_M'],
            hole=0.65,
            hovertemplate='<b>%{label}</b><br>KSh %{value:,.2f}M<extra></extra>'
        )])
        fig1.update_layout(title="Sales by Channel (L1) - Millions KSh", height=500)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Sales by SHIFT
        shift_sales = filtered_df.groupby('SHIFT', as_index=False)['NET_SALES'].sum()
        shift_sales['PCT'] = shift_sales['NET_SALES'] / shift_sales['NET_SALES'].sum() * 100

        fig2 = go.Figure(data=[go.Pie(
            labels=[f"{row['SHIFT']} ({row['PCT']:.1f}%)" for _, row in shift_sales.iterrows()],
            values=shift_sales['NET_SALES'],
            hole=0.65
        )])
        fig2.update_layout(title="Sales Distribution by Shift", height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # Store Sales Summary Table
    st.subheader("Store Performance Summary")
    store_summary = filtered_df.groupby('STORE_NAME').agg({
        'NET_SALES': 'sum',
        'GROSS_SALES': 'sum',
        'CUST_CODE': 'nunique'
    }).reset_index()
    store_summary.columns = ['STORE_NAME', 'NET_SALES', 'GROSS_SALES', 'Unique_Receipts']
    store_summary['Avg_Basket_Value'] = store_summary['GROSS_SALES'] / store_summary['Unique_Receipts']
    store_summary = store_summary.sort_values('GROSS_SALES', ascending=False)

    st.dataframe(
        store_summary.style.format({
            'NET_SALES': '{:,.0f}',
            'GROSS_SALES': '{:,.0f}',
            'Unique_Receipts': '{:,.0f}',
            'Avg_Basket_Value': '{:,.2f}'
        }),
        use_container_width=True,
        height=500
    )

# ====================== TAB 2: CUSTOMER TRAFFIC & TILLS ======================
with tab2:
    st.header("Customer Traffic & Till Activity")

    # Customer Traffic Heatmap (30-min)
    st.subheader("Customer Traffic Heatmap (30-min intervals)")
    # (You can keep or simplify the complex heatmap logic here)
    # For brevity, here's a simplified version:

    filtered_df['TIME_INTERVAL'] = filtered_df['TRN_DATE'].dt.floor('30T')
    traffic = filtered_df.groupby(['STORE_NAME', filtered_df['TIME_INTERVAL'].dt.time])['CUST_CODE'].nunique().reset_index()
    traffic_pivot = traffic.pivot(index='STORE_NAME', columns='TIME_INTERVAL', values='CUST_CODE').fillna(0)

    fig_traffic = px.imshow(
        traffic_pivot,
        labels=dict(x="Time of Day", y="Store", color="Receipts"),
        color_continuous_scale='Reds',
        title="Customer Traffic by Store & Time"
    )
    st.plotly_chart(fig_traffic, use_container_width=True)

    # Active Tills
    st.subheader("Active Tills per Store")
    till_activity = filtered_df.groupby(['STORE_NAME', filtered_df['TRN_DATE'].dt.time])['Till_Code'].nunique().reset_index()
    st.dataframe(till_activity, use_container_width=True)

# ====================== TAB 3: BASKETS & PRODUCTS ======================
with tab3:
    st.header("Baskets & Product Performance")

    col_a, col_b = st.columns(2)

    with col_a:
        top_items = (
            filtered_df.groupby('ITEM_NAME')
            .agg(Baskets=('CUST_CODE', 'nunique'), QTY=('QTY', 'sum'), NET_SALES=('NET_SALES', 'sum'))
            .sort_values('Baskets', ascending=False)
            .head(20)
        )
        st.subheader("Top 20 Items by Baskets")
        st.dataframe(top_items.style.format({'QTY': '{:,.0f}', 'NET_SALES': '{:,.0f}'}))

    with col_b:
        category_sales = filtered_df.groupby('CATEGORY')['NET_SALES'].sum().sort_values(ascending=False)
        fig_cat = px.bar(
            x=category_sales.values,
            y=category_sales.index,
            orientation='h',
            title="Sales by Category"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

# ====================== TAB 4: LOYALTY & REFUNDS ======================
with tab4:
    st.header("Loyalty & Refunds")

    if 'LOYALTY_CUSTOMER_CODE' in filtered_df.columns:
        loyal = filtered_df[filtered_df['LOYALTY_CUSTOMER_CODE'].notna() & (filtered_df['LOYALTY_CUSTOMER_CODE'] != '')]
        st.metric("Loyalty Customers", f"{loyal['LOYALTY_CUSTOMER_CODE'].nunique():,}")

    # Refunds
    refunds = filtered_df[filtered_df['NET_SALES'] < 0]
    st.subheader(f"Refunds / Negative Sales: {len(refunds):,} transactions")
    if not refunds.empty:
        refund_summary = refunds.groupby('STORE_NAME')['NET_SALES'].sum().abs()
        st.bar_chart(refund_summary)

# ====================== TAB 5: RAW SUMMARY TABLES ======================
with tab5:
    st.header("Raw Summary Tables")
    
    if st.checkbox("Show Full Data Sample"):
        st.dataframe(filtered_df.head(1000), use_container_width=True)

    st.subheader("Store-wise Sales Summary")
    summary_table = filtered_df.groupby('STORE_NAME').agg({
        'NET_SALES': 'sum',
        'GROSS_SALES': 'sum',
        'CUST_CODE': 'nunique',
        'ITEM_CODE': 'nunique'
    }).round(2)
    summary_table.columns = ['Net Sales', 'Gross Sales', 'Unique Receipts', 'Unique Items']
    st.dataframe(summary_table.style.format({
        'Net Sales': '{:,.0f}',
        'Gross Sales': '{:,.0f}',
        'Unique Receipts': '{:,.0f}',
        'Unique Items': '{:,.0f}'
    }), use_container_width=True)

# ====================== FOOTER ======================
st.caption("Daily Deck Dashboard | Built with Streamlit | Data: 2026-03-22")
