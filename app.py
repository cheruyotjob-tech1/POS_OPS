# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Daily Deck Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Daily Deck Dashboard")
st.markdown("**POS Transaction Analysis • 2026-03-22**")

# ====================== DATA LOADING ======================
@st.cache_data(ttl=3600)
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(contents))

def smart_load():
    st.sidebar.markdown("### 📁 Data Source")
    uploaded = st.sidebar.file_uploader(
        "Upload DAILY_POS_TRN_ITEMS_*.parquet", 
        type=["parquet"]
    )
    
    if uploaded is not None:
        with st.spinner("Loading uploaded parquet file..."):
            df = load_uploaded_file(uploaded.getvalue())
            st.sidebar.success(f"✅ Loaded: {uploaded.name}")
            return df
    else:
        st.sidebar.info("👆 Please upload the parquet file to begin")
        st.stop()

# Load data
df = smart_load()

# ====================== DATA CLEANING ======================
with st.spinner("Cleaning and preparing data..."):
    # Dates
    df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
    
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
        df['GROSS_SALES'] = df.get('NET_SALES', 0) + df.get('VAT_AMT', 0)

    # CUST_CODE (Unique Receipt)
    for col in ['STORE_CODE', 'TILL', 'SESSION', 'RCT']:
        df[col] = df[col].astype(str).fillna('').str.strip()
    
    df['CUST_CODE'] = df['STORE_CODE'] + '-' + df['TILL'] + '-' + df['SESSION'] + '-' + df['RCT']
    
    # Till_Code & Cashier
    df['Till_Code'] = df['TILL'].astype(str) + '-' + df['STORE_CODE'].astype(str)
    if 'CASHIER' in df.columns:
        df['CASHIER-COUNT'] = df['CASHIER'].astype(str).str.strip() + ' @ ' + df['STORE_NAME'].astype(str)

st.success(f"✅ Data ready! **{len(df):,} rows** × **{len(df.columns)} columns**")

# ====================== SIDEBAR FILTERS ======================
st.sidebar.markdown("### 🔍 Filters")
selected_stores = st.sidebar.multiselect(
    "Stores", 
    options=sorted(df['STORE_NAME'].unique()),
    default=None
)

if selected_stores:
    df = df[df['STORE_NAME'].isin(selected_stores)].copy()

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sales Overview", 
    "👥 Traffic & Tills", 
    "🛒 Baskets & Products", 
    "💰 Loyalty & Refunds", 
    "📋 Tables"
])

# ====================== TAB 1: SALES ======================
with tab1:
    st.header("Sales Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sales_channel = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum()
        sales_channel['PCT'] = sales_channel['NET_SALES'] / sales_channel['NET_SALES'].sum() * 100
        
        fig = go.Figure(data=[go.Pie(
            labels=[f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}%)" for _, row in sales_channel.iterrows()],
            values=sales_channel['NET_SALES'],
            hole=0.65
        )])
        fig.update_layout(title="Sales by Channel L1", height=480)
        st.plotly_chart(fig, use_container_width=True, key="pie1")
    
    with col2:
        shift_sales = df.groupby('SHIFT')['NET_SALES'].sum()
        fig2 = px.pie(names=shift_sales.index, values=shift_sales.values, hole=0.65,
                      title="Sales by Shift")
        st.plotly_chart(fig2, use_container_width=True, key="pie2")

    # Store Summary
    store_summary = df.groupby('STORE_NAME').agg({
        'NET_SALES': 'sum',
        'GROSS_SALES': 'sum',
        'CUST_CODE': 'nunique'
    }).reset_index()
    store_summary['Avg_Basket'] = store_summary['GROSS_SALES'] / store_summary['CUST_CODE']
    store_summary = store_summary.sort_values('GROSS_SALES', ascending=False)
    
    st.subheader("Store Performance")
    st.dataframe(
        store_summary.style.format({
            'NET_SALES': '{:,.0f}',
            'GROSS_SALES': '{:,.0f}',
            'CUST_CODE': '{:,.0f}',
            'Avg_Basket': '{:,.2f}'
        }),
        width="100%",  # Fixed deprecation
        height=400
    )

# ====================== TAB 2: TRAFFIC & TILLS ======================
with tab2:
    st.header("Customer Traffic & Tills")
    
    # Simple 30-min traffic
    df['TIME_30'] = df['TRN_DATE'].dt.floor('30T').dt.time
    traffic = df.groupby(['STORE_NAME', 'TIME_30'])['CUST_CODE'].nunique().reset_index()
    pivot_traffic = traffic.pivot(index='STORE_NAME', columns='TIME_30', values='CUST_CODE').fillna(0)
    
    fig_traffic = px.imshow(
        pivot_traffic,
        labels=dict(x="Time of Day", y="Store", color="Receipts"),
        color_continuous_scale='Reds',
        title="Customer Traffic Heatmap (30-min slots)"
    )
    st.plotly_chart(fig_traffic, width="100%")

# ====================== TAB 3: BASKETS & PRODUCTS ======================
with tab3:
    st.header("Baskets & Product Performance")
    
    top_items = (
        df.groupby('ITEM_NAME')
        .agg(Baskets=('CUST_CODE', 'nunique'),
             Qty=('QTY', 'sum'),
             Sales=('NET_SALES', 'sum'))
        .sort_values('Baskets', ascending=False)
        .head(15)
    )
    
    st.subheader("Top 15 Items by Number of Baskets")
    st.dataframe(top_items.style.format({'Qty': '{:,.0f}', 'Sales': '{:,.0f}'}), width="100%")

# ====================== TAB 4: LOYALTY & REFUNDS ======================
with tab4:
    st.header("Loyalty & Refunds")
    
    refunds = df[df['NET_SALES'] < 0]
    st.metric("Negative Transactions (Refunds)", f"{len(refunds):,}")
    
    if not refunds.empty:
        refund_by_store = refunds.groupby('STORE_NAME')['NET_SALES'].sum().abs()
        st.bar_chart(refund_by_store)

# ====================== TAB 5: TABLES ======================
with tab5:
    st.header("Summary Tables")
    st.dataframe(
        df.groupby('STORE_NAME')[['NET_SALES', 'GROSS_SALES']].sum()
          .style.format('{:,.0f}'),
        width="100%"
    )

st.caption("Daily Deck Dashboard | Fixed for Streamlit Cloud • 2026-03-25")

# Optional: Download cleaned data
@st.cache_data
def convert_df(df):
    return df.to_parquet(index=False)

if st.button("📥 Download Cleaned Parquet"):
    parquet_file = convert_df(df)
    st.download_button(
        label="Download parquet",
        data=parquet_file,
        file_name="cleaned_daily_pos.parquet",
        mime="application/octet-stream"
    )
