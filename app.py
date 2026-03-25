import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

st.set_page_config(page_title="Quickmart Daily Deck", layout="wide", page_icon="🛒")

st.title("🛒 Quickmart Daily Deck")
st.markdown("**POS Intelligence Platform** — Upload your daily parquet files")

# ====================== FILE UPLOAD ======================
st.sidebar.header("📤 Upload Your Parquet Files")
uploaded_files = st.sidebar.file_uploader(
    "Select all DAILY_POS_TRN_ITEMS_*.parquet files from your PC",
    type=["parquet"],
    accept_multiple_files=True,
    help="You can select multiple files at once (Ctrl/Cmd + click)"
)

if uploaded_files:
    st.sidebar.success(f"✅ Uploaded {len(uploaded_files)} parquet files")

    if st.sidebar.button("🚀 Load & Merge Selected Files", type="primary"):
        with st.spinner("Merging parquet files... This may take a moment for large files"):
            dfs = []
            for uploaded_file in uploaded_files:
                try:
                    df_temp = pd.read_parquet(io.BytesIO(uploaded_file.read()))
                    dfs.append(df_temp)
                    st.sidebar.write(f"Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                
                # Cleaning (same as your original notebook)
                df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
                numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

                st.session_state["df"] = df
                st.success(f"✅ Successfully merged {len(dfs)} files → **{df.shape[0]:,} transactions**")
            else:
                st.error("Failed to load any files")
else:
    st.info("👈 **Upload your parquet files from your PC** using the sidebar")

# ====================== TABS (All sections from your notebook) ======================
if "df" not in st.session_state:
    st.stop()

df = st.session_state["df"]

tab1, tab2, tab3, tab4 = st.tabs([
    "🧺 SKU Basket Analysis",
    "👥 Loyalty Overview",
    "🏷️ Pricing Intelligence",
    "🔄 Refunds & Voids"
])

# ====================== TAB 1: SKU Basket ======================
with tab1:
    st.header("SKU Basket Composition")
    # Create nice selectable list
    df['SKU_Display'] = df['ITEM_CODE'].astype(str) + " • " + df['ITEM_NAME'].astype(str)
    selected_sku = st.selectbox("Select an Item", options=sorted(df['SKU_Display'].unique()))
    
    if selected_sku:
        code = selected_sku.split(" • ")[0]
        item_df = df[df['ITEM_CODE'].astype(str) == code].copy()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            summary = item_df.groupby('STORE_NAME').agg(
                Total_Transactions=('CUST_CODE', 'nunique'),
                Total_QTY=('QTY', 'sum'),
                Total_Sales=('NET_SALES', 'sum')
            ).round(2).reset_index()
            st.dataframe(summary, use_container_width=True)
        
        with col2:
            fig = px.bar(summary, x='STORE_NAME', y='Total_QTY',
                         title=f"Quantity Sold by Store - {selected_sku}",
                         color='Total_Sales', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# ====================== TAB 2: Loyalty ======================
with tab2:
    st.header("Loyalty Overview")
    sub1, sub2, sub3 = st.tabs(["🌍 Global Overview", "🏬 By Branch", "👤 Customer Drilldown"])
    
    with sub1:
        st.subheader("Loyal Customers with >1 Basket")
        loyal = df[df['LOYALTY_CUSTOMER_CODE'].notna()].groupby(['STORE_NAME', 'LOYALTY_CUSTOMER_CODE']).agg(
            Baskets=('CUST_CODE', 'nunique'),
            Total_Value=('NET_SALES', 'sum')
        ).reset_index()
        multi_loyal = loyal[loyal['Baskets'] > 1].groupby('STORE_NAME').agg(
            Loyal_Customers=('LOYALTY_CUSTOMER_CODE', 'nunique'),
            Total_Baskets=('Baskets', 'sum'),
            Total_Value=('Total_Value', 'sum')
        ).reset_index()
        st.dataframe(multi_loyal, use_container_width=True)
    
    with sub2:
        st.subheader("Branch Level Loyalty")
        selected_branch = st.selectbox("Choose Branch", df['STORE_NAME'].unique(), key="loyal_branch")
        branch_loyal = df[(df['STORE_NAME'] == selected_branch) & df['LOYALTY_CUSTOMER_CODE'].notna()]
        if not branch_loyal.empty:
            per_cust = branch_loyal.groupby('LOYALTY_CUSTOMER_CODE').agg(
                Baskets=('CUST_CODE', 'nunique'),
                Total_Value=('NET_SALES', 'sum')
            ).reset_index()
            st.dataframe(per_cust[per_cust['Baskets'] > 1], use_container_width=True)
    
    with sub3:
        st.subheader("Single Customer View")
        cust_input = st.text_input("Enter Loyalty Customer Code (e.g. 041018044)")
        if cust_input:
            cust_data = df[df['LOYALTY_CUSTOMER_CODE'].astype(str) == cust_input.strip()]
            if not cust_data.empty:
                st.dataframe(cust_data[['STORE_NAME', 'TRN_DATE', 'NET_SALES', 'ITEM_NAME']], use_container_width=True)
            else:
                st.warning("Customer not found")

# ====================== TAB 3: Pricing ======================
with tab3:
    st.header("Pricing Intelligence")
    st.subheader("SKUs Sold at Multiple Prices")
    
    # Group by store + item + date
    price_data = df.groupby(['STORE_NAME', df['TRN_DATE'].dt.date, 'ITEM_CODE', 'ITEM_NAME']).agg(
        Num_Prices=('SP_PRE_VAT', 'nunique'),
        Min_Price=('SP_PRE_VAT', 'min'),
        Max_Price=('SP_PRE_VAT', 'max'),
        Total_QTY=('QTY', 'sum')
    ).reset_index()
    
    multi_price = price_data[price_data['Num_Prices'] > 1].copy()
    multi_price['Price_Spread'] = multi_price['Max_Price'] - multi_price['Min_Price']
    multi_price['Value_Impact'] = multi_price['Price_Spread'] * multi_price['Total_QTY']
    
    summary = multi_price.groupby('STORE_NAME').agg(
        Affected_SKUs=('ITEM_CODE', 'nunique'),
        Total_Impact=('Value_Impact', 'sum'),
        Avg_Spread=('Price_Spread', 'mean')
    ).round(2).reset_index()
    
    st.dataframe(summary, use_container_width=True)
    
    selected_store = st.selectbox("View Detailed Pricing per Branch", df['STORE_NAME'].unique(), key="price_store")
    detail = multi_price[multi_price['STORE_NAME'] == selected_store]
    if not detail.empty:
        st.dataframe(detail, use_container_width=True)

# ====================== TAB 4: Refunds ======================
with tab4:
    st.header("Refunds & Voids")
    neg_sales = df[df['NET_SALES'] < 0].copy()
    
    if not neg_sales.empty:
        neg_sales['Sale_Type'] = neg_sales['CAP_CUSTOMER_CODE'].apply(
            lambda x: 'On_account sales' if pd.notna(x) and x != '' else 'General sales'
        )
        
        refund_summary = neg_sales.groupby(['STORE_NAME', 'Sale_Type']).agg(
            Negative_Value=('NET_SALES', 'sum'),
            Transaction_Count=('NET_SALES', 'count')
        ).round(2).reset_index()
        
        st.dataframe(refund_summary, use_container_width=True)
        
        total_refund = neg_sales['NET_SALES'].sum()
        st.metric("Total Negative Sales (Refunds/Voids)", f"KSh {total_refund:,.0f}")
    else:
        st.info("No negative sales found in the loaded data.")

st.caption("✅ Full app covering all notebook sections • Built for local PC parquet files")
