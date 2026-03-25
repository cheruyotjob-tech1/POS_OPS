import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os
from datetime import datetime

st.set_page_config(page_title="Quickmart Daily Deck", layout="wide", page_icon="🛒")

st.title("🛒 Quickmart Daily Deck")
st.subheader("POS Intelligence Platform • Real Parquet + Date Range")

# ====================== SIDEBAR ======================
st.sidebar.header("📂 Data Source")
data_folder = st.sidebar.text_input("Folder containing parquet files", value=".", help="e.g. /content/ or ./data/")

if st.sidebar.button("🔍 Scan for Daily Files", type="secondary"):
    files = sorted(glob.glob(os.path.join(data_folder, "DAILY_POS_TRN_ITEMS_*.parquet")))
    if files:
        dates = [os.path.basename(f).split("_")[-1].split(".parquet")[0] for f in files]
        st.session_state["available_files"] = dict(zip(dates, files))
        st.sidebar.success(f"Found {len(files)} files")
    else:
        st.sidebar.error("No matching parquet files found")

st.sidebar.header("📅 Analysis Period")
if "available_files" in st.session_state:
    dates = sorted(st.session_state["available_files"].keys())
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=datetime(2026, 3, 20), min_value=datetime.strptime(min(dates), "%Y-%m-%d"))
    end_date = col2.date_input("End", value=datetime(2026, 3, 22), max_value=datetime.strptime(max(dates), "%Y-%m-%d"))

    if st.sidebar.button("🚀 Load & Merge Data", type="primary"):
        with st.spinner("Merging selected daily parquet files..."):
            files_to_load = [st.session_state["available_files"][d] for d in dates 
                            if str(start_date) <= d <= str(end_date)]
            if files_to_load:
                dfs = []
                for f in files_to_load:
                    temp = pd.read_parquet(f)
                    dfs.append(temp)
                df = pd.concat(dfs, ignore_index=True)

                # Cleaning (same as notebook)
                df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
                numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

                st.session_state["df"] = df
                st.sidebar.success(f"✅ Loaded {df.shape[0]:,} rows from {len(files_to_load)} days")
            else:
                st.sidebar.error("No files in selected range")
else:
    st.sidebar.info("Scan folder first")

# ====================== TABS ======================
if "df" not in st.session_state:
    st.info("👈 Use the sidebar to load data first")
    st.stop()

df = st.session_state["df"]

tab1, tab2, tab3, tab4 = st.tabs([
    "🧺 SKU Basket Analysis",
    "👥 Loyalty Overview",
    "🏷️ Pricing Intelligence",
    "🔄 Refunds & Voids"
])

# ====================== TAB 1: SKU BASKET ======================
with tab1:
    st.header("SKU Basket Composition")
    sku_list = (df['ITEM_CODE'].astype(str) + " • " + df['ITEM_NAME'].astype(str)).unique()
    selected = st.selectbox("Select Item", options=sorted(sku_list), key="sku_select")
    
    if selected:
        code = selected.split(" • ")[0]
        item_data = df[df['ITEM_CODE'].astype(str) == code].copy()
        
        # Basket-like summary (per store)
        summary = item_data.groupby('STORE_NAME').agg(
            Baskets=('CUST_CODE', 'nunique'),
            Total_QTY=('QTY', 'sum'),
            Total_Net=('NET_SALES', 'sum')
        ).reset_index()
        
        st.dataframe(summary, use_container_width=True)
        
        fig = px.bar(summary, x='STORE_NAME', y='Total_QTY', 
                     title=f"Quantity Sold by Store – {selected}",
                     color='Total_Net')
        st.plotly_chart(fig, use_container_width=True)

# ====================== TAB 2: LOYALTY ======================
with tab2:
    st.header("Loyalty Overview")
    sub1, sub2, sub3 = st.tabs(["Global", "By Branch", "Customer Drilldown"])
    
    with sub1:
        st.subheader("Global Loyalty (Customers with >1 basket)")
        loyal = df[df['LOYALTY_CUSTOMER_CODE'].notna()].groupby(
            ['STORE_NAME', 'LOYALTY_CUSTOMER_CODE']
        ).agg(Baskets=('CUST_CODE', 'nunique'), Value=('NET_SALES', 'sum')).reset_index()
        multi = loyal[loyal['Baskets'] > 1].groupby('STORE_NAME').agg(
            Loyal_Customers=('LOYALTY_CUSTOMER_CODE', 'nunique'),
            Total_Baskets=('Baskets', 'sum'),
            Total_Value=('Value', 'sum')
        ).reset_index()
        st.dataframe(multi, use_container_width=True)
    
    with sub2:
        st.subheader("Branch Loyalty")
        branch = st.selectbox("Select Branch", options=df['STORE_NAME'].unique(), key="branch_loyal")
        branch_data = df[(df['STORE_NAME'] == branch) & df['LOYALTY_CUSTOMER_CODE'].notna()]
        if not branch_data.empty:
            per_cust = branch_data.groupby('LOYALTY_CUSTOMER_CODE').agg(
                Baskets=('CUST_CODE', 'nunique'),
                Value=('NET_SALES', 'sum')
            ).reset_index()
            st.dataframe(per_cust[per_cust['Baskets'] > 1], use_container_width=True)
    
    with sub3:
        st.subheader("Customer Deep Dive")
        cust_code = st.text_input("Enter Loyalty Customer Code")
        if cust_code and cust_code in df['LOYALTY_CUSTOMER_CODE'].astype(str).values:
            cust_df = df[df['LOYALTY_CUSTOMER_CODE'].astype(str) == cust_code]
            st.dataframe(cust_df[['STORE_NAME', 'TRN_DATE', 'NET_SALES']], use_container_width=True)

# ====================== TAB 3: PRICING ======================
with tab3:
    st.header("Pricing Intelligence")
    st.subheader("Items Sold at Multiple Prices")
    
    price_group = df.groupby(['STORE_NAME', 'DATE' if 'DATE' in df.columns else df['TRN_DATE'].dt.date, 
                              'ITEM_CODE', 'ITEM_NAME']).agg(
        Num_Prices=('SP_PRE_VAT', 'nunique'),
        Min_Price=('SP_PRE_VAT', 'min'),
        Max_Price=('SP_PRE_VAT', 'max'),
        Total_QTY=('QTY', 'sum')
    ).reset_index()
    
    multi_price = price_group[price_group['Num_Prices'] > 1].copy()
    multi_price['Spread'] = multi_price['Max_Price'] - multi_price['Min_Price']
    multi_price['Value_Impact'] = multi_price['Spread'] * multi_price['Total_QTY']
    
    summary = multi_price.groupby('STORE_NAME').agg(
        Affected_SKUs=('ITEM_CODE', 'nunique'),
        Total_Impact=('Value_Impact', 'sum'),
        Avg_Spread=('Spread', 'mean')
    ).reset_index()
    
    st.dataframe(summary, use_container_width=True)
    
    branch_pr = st.selectbox("Branch Pricing Drilldown", options=df['STORE_NAME'].unique(), key="branch_price")
    drill = multi_price[multi_price['STORE_NAME'] == branch_pr]
    if not drill.empty:
        st.dataframe(drill, use_container_width=True)

# ====================== TAB 4: REFUNDS ======================
with tab4:
    st.header("Refunds & Voids")
    neg = df[df['NET_SALES'] < 0].copy()
    neg['Sale_Type'] = neg['CAP_CUSTOMER_CODE'].fillna('').apply(
        lambda x: 'On_account sales' if x != '' else 'General sales'
    )
    
    summary = neg.groupby(['STORE_NAME', 'Sale_Type']).agg(
        Total_Neg_Value=('NET_SALES', 'sum'),
        Count=('NET_SALES', 'count')
    ).reset_index()
    
    st.dataframe(summary, use_container_width=True)
    
    total_neg = neg['NET_SALES'].sum()
    st.metric("Overall Negative Value", f"KSh {total_neg:,.0f}", delta=None)

# Footer
st.caption("✅ Fully interactive app built from your original notebook • Real multi-day Parquet merging")
