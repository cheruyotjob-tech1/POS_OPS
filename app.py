import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from io import BytesIO

st.set_page_config(layout="wide", page_title="DailyDeck")

# ====================== DATA LOADING ======================
@st.cache_data
def load_uploaded_file(contents: bytes) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(contents))


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def smart_load():
    st.sidebar.markdown("### Upload Data")
    uploaded = st.sidebar.file_uploader("Upload DAILY_POS_TRN_ITEMS parquet", type=['parquet'])
    
    if uploaded is not None:
        with st.spinner("Loading uploaded parquet..."):
            df = load_uploaded_file(uploaded.getvalue())
        st.sidebar.success("✅ Uploaded file loaded successfully")
        return df

    # Fallback (optional)
    default_path = "/content/DAILY_POS_TRN_ITEMS_2026-03-22.parquet"
    try:
        with st.spinner("Loading default parquet..."):
            df = load_parquet(default_path)
        st.sidebar.info("Loaded default parquet")
        return df
    except:
        st.sidebar.warning("No default file found. Please upload a parquet file.")
        return None


# ====================== CLEANING & DERIVED COLUMNS ======================
@st.cache_data
def clean_and_derive(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    
    df = raw_df.copy()

    # String normalization
    str_cols = ['STORE_CODE', 'TILL', 'SESSION', 'RCT', 'STORE_NAME', 'CASHIER', 
                'ITEM_CODE', 'ITEM_NAME', 'DEPARTMENT', 'CATEGORY', 'SUPPLIER_NAME',
                'SALES_CHANNEL_L1', 'SALES_CHANNEL_L2', 'SHIFT']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()

    # Date handling
    if 'TRN_DATE' in df.columns:
        df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
        df = df.dropna(subset=['TRN_DATE']).copy()
        df['DATE'] = df['TRN_DATE'].dt.date
        df['TIME_INTERVAL'] = df['TRN_DATE'].dt.floor('30min')
        df['TIME_ONLY'] = df['TIME_INTERVAL'].dt.time

    if 'ZED_DATE' in df.columns:
        df['ZED_DATE'] = pd.to_datetime(df['ZED_DATE'], errors='coerce')

    # Numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '', regex=False).str.strip(),
                errors='coerce'
            ).fillna(0)

    # Derived columns
    if all(c in df.columns for c in ['STORE_CODE', 'TILL', 'SESSION', 'RCT']):
        df['CUST_CODE'] = (df['STORE_CODE'].astype(str) + '-' +
                          df['TILL'].astype(str) + '-' +
                          df['SESSION'].astype(str) + '-' +
                          df['RCT'].astype(str))
    else:
        df['CUST_CODE'] = ''

    if 'STORE_CODE' in df.columns and 'TILL' in df.columns:
        df['Till_Code'] = df['TILL'].astype(str) + '-' + df['STORE_CODE'].astype(str)

    if 'STORE_NAME' in df.columns and 'CASHIER' in df.columns:
        df['CASHIER-COUNT'] = df['CASHIER'].astype(str) + '-' + df['STORE_NAME'].astype(str)

    if 'SHIFT' in df.columns:
        df['Shift_Bucket'] = np.where(
            df['SHIFT'].str.upper().str.contains('NIGHT', na=False), 'Night', 'Day'
        )

    # GROSS_SALES
    df['GROSS_SALES'] = df.get('NET_SALES', 0) + df.get('VAT_AMT', 0)

    return df


# ====================== HELPER FUNCTIONS ======================
def format_and_display(df: pd.DataFrame, numeric_cols=None, index_col=None, total_label='TOTAL'):
    if df is None or df.empty:
        st.dataframe(df)
        return

    df_display = df.copy()
    if numeric_cols is None:
        numeric_cols = list(df_display.select_dtypes(include=[np.number]).columns)

    # Add total row
    totals = {}
    for col in df_display.columns:
        if col in numeric_cols:
            totals[col] = df_display[col].sum()
        else:
            totals[col] = ''

    totals[index_col or df_display.columns[0]] = total_label
    tot_df = pd.DataFrame([totals])
    appended = pd.concat([df_display, tot_df], ignore_index=True)

    # Format numbers
    for col in numeric_cols:
        if col in appended.columns:
            series = appended[col].dropna()
            if len(series) > 0 and np.allclose(series.fillna(0).round(0), series.fillna(0)):
                appended[col] = appended[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else '')
            else:
                appended[col] = appended[col].map(lambda x: f"{float(x):,.2f}" if pd.notna(x) else '')

    st.dataframe(appended, width="stretch")


def donut_from_agg(df_agg, label_col, value_col, title, hole=0.55, value_is_millions=False):
    labels = df_agg[label_col].astype(str).tolist()
    vals = df_agg[value_col].astype(float).tolist()
    
    if value_is_millions:
        values_for_plot = [v / 1_000_000 for v in vals]
        hover = 'KSh %{value:,.2f} M'
    else:
        values_for_plot = vals
        hover = 'KSh %{value:,.2f}'

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values_for_plot,
        hole=hole,
        hovertemplate='<b>%{label}</b><br>' + hover + '<extra></extra>',
        marker=dict(line=dict(color='white', width=1))
    )])
    fig.update_layout(title=title)
    return fig


# ====================== MAIN APP ======================
def main():
    st.title("DailyDeck: The Story Behind the Numbers")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None

    raw_df = smart_load()
    if raw_df is None:
        st.stop()

    with st.spinner("Cleaning and preparing data..."):
        df = clean_and_derive(raw_df)
        st.session_state.df = df

    if df.empty:
        st.error("No data available after cleaning.")
        st.stop()

    st.sidebar.success(f"✅ Data loaded: {len(df):,} rows")

    # Sidebar navigation
    section = st.sidebar.selectbox(
        "Choose Section",
        ["SALES", "OPERATIONS", "INSIGHTS"]
    )

    if section == "SALES":
        sales_options = [
            "Global Sales Overview",
            "Sales by Channel L2",
            "Sales by Shift",
            "Night vs Day Ratio",
            "Global Day vs Night",
            "2nd Highest Channel Share",
            "Bottom 30 - 2nd Highest Channel",
            "Stores Sales Summary"
        ]
        choice = st.sidebar.selectbox("Sales Analysis", sales_options)

        if choice == "Global Sales Overview":
            sales_global_overview(df)
        elif choice == "Sales by Channel L2":
            sales_by_channel_l2(df)
        elif choice == "Sales by Shift":
            sales_by_shift(df)
        elif choice == "Night vs Day Ratio":
            night_vs_day_ratio(df)
        elif choice == "Global Day vs Night":
            global_day_vs_night(df)
        elif choice == "2nd Highest Channel Share":
            second_highest_channel_share(df)
        elif choice == "Bottom 30 - 2nd Highest Channel":
            bottom_30_2nd_highest(df)
        elif choice == "Stores Sales Summary":
            stores_sales_summary(df)

    elif section == "OPERATIONS":
        ops_options = [
            "Customer Traffic (Storewise)",
            "Active Tills During Day",
            "Average Customers per Till"
        ]
        choice = st.sidebar.selectbox("Operations Analysis", ops_options)

        if choice == "Customer Traffic (Storewise)":
            customer_traffic_storewise(df)
        elif choice == "Active Tills During Day":
            active_tills_during_day(df)
        elif choice == "Average Customers per Till":
            avg_customers_per_till(df)

    elif section == "INSIGHTS":
        ins_options = [
            "Global Category Overview - Sales",
            "Global Category Overview - Baskets",
            "Supplier Contribution",
            "Branch Comparison",
            "Product Performance",
            "Global Loyalty Overview",
            "Global Pricing Overview",
            "Global Refunds Overview"
        ]
        choice = st.sidebar.selectbox("Insights", ins_options)

        if choice == "Global Category Overview - Sales":
            global_category_overview_sales(df)
        elif choice == "Global Category Overview - Baskets":
            global_category_overview_baskets(df)
        elif choice == "Supplier Contribution":
            supplier_contribution(df)
        elif choice == "Branch Comparison":
            branch_comparison(df)
        elif choice == "Product Performance":
            product_performance(df)
        elif choice == "Global Loyalty Overview":
            global_loyalty_overview(df)
        elif choice == "Global Pricing Overview":
            global_pricing_overview(df)
        elif choice == "Global Refunds Overview":
            global_refunds_overview(df)


# ====================== SALES FUNCTIONS ======================
def sales_global_overview(df):
    st.header("Global Sales Overview")
    if 'SALES_CHANNEL_L1' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L1 or NET_SALES")
        return
    g = df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    fig = donut_from_agg(g, 'SALES_CHANNEL_L1', 'NET_SALES', "Sales Channel Type — Global Overview", value_is_millions=True)
    st.plotly_chart(fig, width="stretch")
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L1')


def sales_by_channel_l2(df):
    st.header("Global Net Sales by Sales Channel L2")
    if 'SALES_CHANNEL_L2' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SALES_CHANNEL_L2 or NET_SALES")
        return
    g = df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    fig = donut_from_agg(g, 'SALES_CHANNEL_L2', 'NET_SALES', "Global Net Sales Distribution by Sales Mode", value_is_millions=True)
    st.plotly_chart(fig, width="stretch")
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='SALES_CHANNEL_L2')


def sales_by_shift(df):
    st.header("Global Net Sales Distribution by SHIFT")
    if 'SHIFT' not in df.columns or 'NET_SALES' not in df.columns:
        st.warning("Missing SHIFT or NET_SALES")
        return
    g = df.groupby('SHIFT', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    g['PCT'] = (100 * g['NET_SALES'] / g['NET_SALES'].sum()).round(1)
    fig = go.Figure(data=[go.Pie(labels=g['SHIFT'], values=g['NET_SALES'], hole=0.65)])
    fig.update_layout(title="Sales Distribution by Shift")
    st.plotly_chart(fig, width="stretch")
    format_and_display(g, numeric_cols=['NET_SALES', 'PCT'], index_col='SHIFT')


def night_vs_day_ratio(df):
    st.header("Night vs Day Shift Sales Ratio")
    if 'Shift_Bucket' not in df.columns or 'STORE_NAME' not in df.columns:
        st.warning("Missing Shift_Bucket or STORE_NAME")
        return
    stores_with_night = df[df['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    if len(stores_with_night) == 0:
        st.info("No stores with Night shifts found")
        return
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)]
    ratio = df_nd.groupby(['STORE_NAME', 'Shift_Bucket'])['NET_SALES'].sum().reset_index()
    ratio['STORE_TOTAL'] = ratio.groupby('STORE_NAME')['NET_SALES'].transform('sum')
    ratio['PCT'] = (100 * ratio['NET_SALES'] / ratio['STORE_TOTAL']).round(1)
    pivot = ratio.pivot(index='STORE_NAME', columns='Shift_Bucket', values='PCT').fillna(0)
    pivot = pivot.sort_values('Night', ascending=False)
    st.dataframe(pivot, width="stretch")


def global_day_vs_night(df):
    st.header("Global Day vs Night Sales")
    if 'Shift_Bucket' not in df.columns:
        st.warning("Missing Shift_Bucket")
        return
    stores_with_night = df[df['Shift_Bucket'] == 'Night']['STORE_NAME'].unique()
    df_nd = df[df['STORE_NAME'].isin(stores_with_night)]
    if df_nd.empty:
        st.info("No night shift stores")
        return
    agg = df_nd.groupby('Shift_Bucket')['NET_SALES'].sum().reset_index()
    agg['PCT'] = (100 * agg['NET_SALES'] / agg['NET_SALES'].sum()).round(1)
    fig = go.Figure(go.Pie(labels=agg['Shift_Bucket'], values=agg['NET_SALES'], hole=0.65))
    st.plotly_chart(fig, width="stretch")
    format_and_display(agg, numeric_cols=['NET_SALES', 'PCT'], index_col='Shift_Bucket')


def second_highest_channel_share(df):
    st.header("2nd Highest Channel Share per Store")
    if not all(col in df.columns for col in ['STORE_NAME', 'SALES_CHANNEL_L1', 'NET_SALES']):
        st.warning("Missing required columns")
        return
    # Implementation simplified - add full logic if needed
    st.info("2nd Highest Channel Share analysis - Coming soon")


def bottom_30_2nd_highest(df):
    st.header("Bottom 30 Stores by 2nd Highest Channel")
    st.info("Bottom 30 analysis - Coming soon")


def stores_sales_summary(df):
    st.header("Stores Sales Summary")
    if 'STORE_NAME' not in df.columns:
        st.warning("Missing STORE_NAME")
        return
    summary = df.groupby('STORE_NAME').agg(
        NET_SALES=('NET_SALES', 'sum'),
        GROSS_SALES=('GROSS_SALES', 'sum'),
        Transactions=('CUST_CODE', 'nunique')
    ).reset_index()
    summary['% Contribution'] = (summary['GROSS_SALES'] / summary['GROSS_SALES'].sum() * 100).round(2)
    format_and_display(summary, numeric_cols=['NET_SALES', 'GROSS_SALES', '% Contribution', 'Transactions'], index_col='STORE_NAME')


# ====================== OPERATIONS FUNCTIONS ======================
def customer_traffic_storewise(df):
    st.header("Customer Traffic Heatmap — Storewise (30-min)")
    st.info("Customer Traffic analysis - Coming soon (requires CUST_CODE)")


def active_tills_during_day(df):
    st.header("Peak Active Tills")
    st.info("Active Tills analysis - Coming soon")


def avg_customers_per_till(df):
    st.header("Average Customers Served per Till")
    st.info("Average Customers per Till - Coming soon")


# ====================== INSIGHTS FUNCTIONS ======================
def global_category_overview_sales(df):
    st.header("Global Category Overview — Sales")
    if 'CATEGORY' not in df.columns:
        st.warning("Missing CATEGORY column")
        return
    g = df.groupby('CATEGORY', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False)
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='CATEGORY')
    fig = px.bar(g.head(15), x='NET_SALES', y='CATEGORY', orientation='h', title="Top 15 Categories by Net Sales")
    st.plotly_chart(fig, width="stretch")


def global_category_overview_baskets(df):
    st.header("Global Category Overview — Baskets")
    if 'CATEGORY' not in df.columns or 'CUST_CODE' not in df.columns:
        st.warning("Missing CATEGORY or CUST_CODE")
        return
    g = df.groupby('CATEGORY')['CUST_CODE'].nunique().reset_index(name='Baskets').sort_values('Baskets', ascending=False)
    format_and_display(g, numeric_cols=['Baskets'], index_col='CATEGORY')


def supplier_contribution(df):
    st.header("Supplier Contribution")
    if 'SUPPLIER_NAME' not in df.columns:
        st.warning("Missing SUPPLIER_NAME")
        return
    g = df.groupby('SUPPLIER_NAME', as_index=False)['NET_SALES'].sum().sort_values('NET_SALES', ascending=False).head(30)
    format_and_display(g, numeric_cols=['NET_SALES'], index_col='SUPPLIER_NAME')


def branch_comparison(df):
    st.header("Branch Comparison")
    st.info("Branch Comparison feature - Coming soon")


def product_performance(df):
    st.header("Product Performance")
    if 'ITEM_CODE' not in df.columns or 'CUST_CODE' not in df.columns:
        st.warning("Missing ITEM_CODE or CUST_CODE")
        return
    st.info("Product Performance analysis - Coming soon")


def global_loyalty_overview(df):
    st.header("Global Loyalty Overview")
    if 'LOYALTY_CUSTOMER_CODE' not in df.columns:
        st.warning("Missing LOYALTY_CUSTOMER_CODE")
        return
    st.info("Loyalty Overview - Coming soon")


def global_pricing_overview(df):
    st.header("Global Pricing Overview (Multi-priced SKUs)")
    st.info("Pricing analysis - Coming soon")


def global_refunds_overview(df):
    st.header("Global Refunds Overview")
    df_neg = df[df['NET_SALES'] < 0].copy()
    if df_neg.empty:
        st.success("No negative (refund) transactions found")
        return
    summary = df_neg.groupby('STORE_NAME').agg(
        Total_Refunds=('NET_SALES', 'sum'),
        Count=('CUST_CODE', 'nunique')
    ).reset_index()
    format_and_display(summary, numeric_cols=['Total_Refunds', 'Count'], index_col='STORE_NAME')


if __name__ == "__main__":
    main()
