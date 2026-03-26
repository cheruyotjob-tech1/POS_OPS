import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
from decimal import Decimal, ROUND_HALF_UP
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG & CACHING
# ============================================
st.set_page_config(
    page_title="Daily Deck Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def load_data(file_path):
    """Load parquet file with caching"""
    df = pd.read_parquet(file_path)
    
    # Convert dates
    df['TRN_DATE'] = pd.to_datetime(df['TRN_DATE'], errors='coerce')
    df['ZED_DATE'] = pd.to_datetime(df['ZED_DATE'], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['QTY', 'CP_PRE_VAT', 'SP_PRE_VAT', 'COST_PRE_VAT', 'NET_SALES', 'VAT_AMT']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived columns
    if 'STORE_CODE' in df.columns and 'TILL' in df.columns:
        for col in ['STORE_CODE', 'TILL', 'SESSION', 'RCT']:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].astype(str).fillna('').str.strip()
        df['CUST_CODE'] = df['STORE_CODE'] + '-' + df['TILL'] + '-' + df['SESSION'] + '-' + df['RCT']
    
    if 'STORE_CODE' in df.columns and 'TILL' in df.columns:
        df['Till_Code'] = df['TILL'] + '-' + df['STORE_CODE']
    
    if 'CASHIER' in df.columns and 'STORE_NAME' in df.columns:
        df['CASHIER-COUNT'] = df['CASHIER'].astype(str).str.strip() + '-' + df['STORE_NAME'].astype(str).str.strip()
    
    df['NET_SALES'] = pd.to_numeric(df['NET_SALES'], errors='coerce').fillna(0)
    df['VAT_AMT'] = pd.to_numeric(df['VAT_AMT'], errors='coerce').fillna(0)
    if 'NET_SALES' in df.columns and 'VAT_AMT' in df.columns:
        df['GROSS_SALES'] = df['NET_SALES'] + df['VAT_AMT']
    
    return df

def _round2(v: float) -> float:
    """Stable currency rounding"""
    return float(Decimal(str(v)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def fmt_int(df_in, cols):
    """Format columns with comma separators"""
    df = df_in.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map('{:,.0f}'.format)
    return df

# ============================================
# SIDEBAR & DATA UPLOAD
# ============================================
st.sidebar.title("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Parquet File", type=['parquet'])

if uploaded_file is None:
    st.info("👈 Please upload a parquet file in the sidebar to get started.")
    st.stop()

df = load_data(uploaded_file)

st.sidebar.success(f"✅ Data loaded: {len(df):,} rows")

# Navigation menu
st.sidebar.title("📑 Navigation")
section = st.sidebar.radio(
    "Select Section:",
    [
        "Overview",
        "Sales Analysis",
        "Operations",
        "Customer Traffic",
        "Loyalty",
        "Pricing",
        "Refunds"
    ]
)

# ============================================
# SECTION 1: OVERVIEW
# ============================================
if section == "Overview":
    st.title("📊 Daily Deck Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Stores", df['STORE_NAME'].nunique())
    with col3:
        st.metric("Total Net Sales", f"KSh {df['NET_SALES'].sum():,.0f}M" if df['NET_SALES'].sum() >= 1e6 else f"KSh {df['NET_SALES'].sum():,.0f}")
    with col4:
        st.metric("Date Range", f"{df['TRN_DATE'].min().date()} to {df['TRN_DATE'].max().date()}")
    
    st.divider()
    
    # Global sales overview
    st.subheader("🌍 Global Sales Overview")
    
    global_sales = (
        df.groupby('SALES_CHANNEL_L1', as_index=False)['NET_SALES']
        .sum()
        .sort_values('NET_SALES', ascending=False)
    )
    global_sales['NET_SALES_M'] = global_sales['NET_SALES'] / 1_000_000
    global_sales['PCT'] = (global_sales['NET_SALES'] / global_sales['NET_SALES'].sum()) * 100
    
    legend_labels = [
        f"{row['SALES_CHANNEL_L1']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)"
        for _, row in global_sales.iterrows()
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=legend_labels,
        values=global_sales['NET_SALES_M'],
        hole=0.65,
        text=[f"{p:.1f}%" for p in global_sales['PCT']],
        textinfo='text',
        textposition='inside',
        marker=dict(
            colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'],
            line=dict(color='white', width=1)
        )
    )])
    
    fig.update_layout(height=600, title_text="Sales Channel Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Store Sales Summary
    st.subheader("🏪 Store Sales Summary")
    
    def _to_num(s):
        return pd.to_numeric(
            pd.Series(s).astype(str).str.replace(r'[,%]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    
    agg = (
        df.groupby('STORE_NAME', as_index=False)
        .agg(
            NET_SALES=('NET_SALES', 'sum'),
            GROSS_SALES=('GROSS_SALES', 'sum') if 'GROSS_SALES' in df.columns else ('NET_SALES', 'sum'),
            Customer_Numbers=('CUST_CODE', 'nunique')
        )
    )
    
    agg['Gross_BV'] = np.where(
        agg['Customer_Numbers'] > 0,
        agg['GROSS_SALES'] / agg['Customer_Numbers'],
        0.0
    )
    
    total_net = agg['NET_SALES'].sum()
    agg['% Contribution'] = np.where(
        total_net > 0,
        (agg['NET_SALES'] / total_net) * 100,
        0.0
    )
    
    agg = agg.sort_values('NET_SALES', ascending=False).reset_index(drop=True)
    
    tot_row = pd.DataFrame({
        'STORE_NAME': ['TOTAL'],
        'NET_SALES': [agg['NET_SALES'].sum()],
        'GROSS_SALES': [agg['GROSS_SALES'].sum()],
        'Customer_Numbers': [agg['Customer_Numbers'].sum()],
        'Gross_BV': [agg['GROSS_SALES'].sum() / agg['Customer_Numbers'].sum() if agg['Customer_Numbers'].sum() > 0 else 0.0],
        '% Contribution': [100.0]
    })
    
    final = pd.concat([tot_row, agg], ignore_index=True)
    
    disp = final.copy()
    disp['NET_SALES'] = disp['NET_SALES'].map('{:,.0f}'.format)
    disp['GROSS_SALES'] = disp['GROSS_SALES'].map('{:,.0f}'.format)
    disp['Customer_Numbers'] = disp['Customer_Numbers'].map('{:,.0f}'.format)
    disp['Gross_BV'] = disp['Gross_BV'].map('{:,.2f}'.format)
    disp['% Contribution'] = disp['% Contribution'].map('{:,.2f}'.format)
    
    disp.insert(0, '#', [''] + list(range(1, len(final))))
    
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ============================================
# SECTION 2: SALES ANALYSIS
# ============================================
elif section == "Sales Analysis":
    st.title("💰 Sales Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Channel Distribution", "Category Analysis", "Supplier Contribution"])
    
    with tab1:
        st.subheader("Sales Channel Distribution")
        
        channel2_sales = (
            df.groupby('SALES_CHANNEL_L2', as_index=False)['NET_SALES']
            .sum()
            .sort_values('NET_SALES', ascending=False)
        )
        
        channel2_sales['NET_SALES_M'] = channel2_sales['NET_SALES'] / 1_000_000
        channel2_sales['PCT'] = (
            channel2_sales['NET_SALES'] / channel2_sales['NET_SALES'].sum() * 100
        )
        
        legend_labels = [
            f"{row['SALES_CHANNEL_L2']} ({row['PCT']:.1f}% | {row['NET_SALES_M']:.1f}M)"
            for _, row in channel2_sales.iterrows()
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=legend_labels,
            values=channel2_sales['NET_SALES_M'],
            hole=0.65,
            text=[f"{p:.1f}%" for p in channel2_sales['PCT']],
            textinfo='text',
            textposition='inside',
            marker=dict(
                colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'],
                line=dict(color='white', width=1)
            )
        )])
        
        total_sales_m = channel2_sales['NET_SALES_M'].sum()
        fig.update_layout(
            height=600,
            title_text="Sales Mode Distribution",
            legend_title_text=f"Sales Mode\nTotal: {total_sales_m:,.1f}M"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Category Contribution by Branch")
        
        dc = df.copy()
        dc['NET_SALES'] = pd.to_numeric(dc['NET_SALES'], errors='coerce').fillna(0)
        
        summary = (
            dc.groupby(['STORE_NAME','CATEGORY'], as_index=False)
            .agg(Total_Sales=('NET_SALES','sum'))
        )
        
        summary['Branch_Total'] = summary.groupby('STORE_NAME')['Total_Sales'].transform('sum')
        summary['Pct_of_Branch'] = (summary['Total_Sales'] / summary['Branch_Total'] * 100)
        
        branches_sorted = sorted(summary['STORE_NAME'].unique())
        
        fig = px.bar(
            summary.sort_values('STORE_NAME'),
            y='STORE_NAME',
            x='Total_Sales',
            color='CATEGORY',
            orientation='h',
            title='Category Contribution by Branch'
        )
        
        fig.update_layout(
            barmode='stack',
            height=max(500, 25*len(branches_sorted)),
            xaxis_title='Total Sales (KSh)',
            yaxis_title='Branch'
        )
        fig.update_yaxes(categoryorder='array', categoryarray=branches_sorted)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Supplier Contribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            supplier = st.selectbox("Select Supplier:", sorted(df['SUPPLIER_NAME'].unique()))
        
        with col2:
            category = st.selectbox("Select Category:", ['ALL'] + sorted(df[df['SUPPLIER_NAME']==supplier]['CATEGORY'].unique()))
        
        with col3:
            department = st.selectbox("Select Department:", ['ALL'] + sorted(df[(df['SUPPLIER_NAME']==supplier) & ((df['CATEGORY']==category) if category!='ALL' else True)]['DEPARTMENT'].unique()))
        
        # Filter data
        scope = df[df['SUPPLIER_NAME'] == supplier].copy()
        if category != 'ALL':
            scope = scope[scope['CATEGORY'] == category]
        if department != 'ALL':
            scope = scope[scope['DEPARTMENT'] == department]
        
        if scope.empty:
            st.warning("No data for selected filters")
        else:
            # Calculate metrics
            total_baskets_all = df.groupby('STORE_NAME')['CUST_CODE'].nunique()
            baskets_with_supplier = scope.groupby('STORE_NAME')['CUST_CODE'].nunique()
            
            tbl = (
                total_baskets_all.to_frame('Total_Baskets_All')
                .merge(baskets_with_supplier.to_frame('Baskets_With_Supplier'), 
                       left_index=True, right_index=True, how='left')
                .fillna({'Baskets_With_Supplier': 0})
                .reset_index()
                .rename(columns={'index':'STORE_NAME'})
            )
            
            tbl['Supplier_Share_%'] = np.where(
                tbl['Total_Baskets_All'] > 0,
                (tbl['Baskets_With_Supplier'] / tbl['Total_Baskets_All'] * 100).round(2),
                0.0
            )
            
            tbl = tbl.sort_values('Supplier_Share_%', ascending=False)
            
            fig = px.bar(
                tbl.sort_values('STORE_NAME'),
                x='Supplier_Share_%',
                y='STORE_NAME',
                orientation='h',
                text='Supplier_Share_%',
                title=f"Supplier Share by Store - {supplier}"
            )
            
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(
                height=max(400, 20*len(tbl)),
                xaxis_title='% of Baskets',
                yaxis_title='Store'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Details Table")
            tbl_disp = tbl.copy()
            tbl_disp['Total_Baskets_All'] = tbl_disp['Total_Baskets_All'].map('{:,}'.format)
            tbl_disp['Baskets_With_Supplier'] = tbl_disp['Baskets_With_Supplier'].map('{:,}'.format)
            tbl_disp['Supplier_Share_%'] = tbl_disp['Supplier_Share_%'].map('{:.2f}%'.format)
            
            st.dataframe(tbl_disp, use_container_width=True, hide_index=True)

# ============================================
# SECTION 3: OPERATIONS
# ============================================
elif section == "Operations":
    st.title("🔧 Operations Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Customer Traffic", "Active Tills", "Cashier Performance", "Till Usage"])
    
    with tab1:
        st.subheader("Customer Traffic Heatmap")
        
        df_traffic = df.copy()
        df_traffic['TRN_DATE'] = pd.to_datetime(df_traffic['TRN_DATE'], errors='coerce')
        df_traffic = df_traffic.dropna(subset=['TRN_DATE'])
        
        df_traffic['TRN_DATE_ONLY'] = df_traffic['TRN_DATE'].dt.date
        df_traffic['TIME_INTERVAL'] = df_traffic['TRN_DATE'].dt.floor('30T')
        df_traffic['TIME_ONLY'] = df_traffic['TIME_INTERVAL'].dt.time
        
        start_time = pd.Timestamp("00:00:00")
        intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        
        first_touch = (
            df_traffic.dropna(subset=['TRN_DATE'])
            .groupby(['STORE_NAME','TRN_DATE_ONLY','CUST_CODE'], as_index=False)['TRN_DATE']
            .min()
        )
        
        first_touch['TIME_INTERVAL'] = first_touch['TRN_DATE'].dt.floor('30T')
        first_touch['TIME_ONLY'] = first_touch['TIME_INTERVAL'].dt.time
        
        counts = (
            first_touch.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE']
            .nunique()
            .reset_index(name='RECEIPT_COUNT')
        )
        
        heatmap = counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='RECEIPT_COUNT').fillna(0)
        
        for t in intervals:
            if t not in heatmap.columns:
                heatmap[t] = 0
        heatmap = heatmap[intervals]
        
        heatmap['TOTAL'] = heatmap.sum(axis=1)
        heatmap = heatmap.sort_values('TOTAL', ascending=False)
        heatmap_matrix = heatmap.drop(columns=['TOTAL'])
        
        colorscale = [
            [0.0,  '#E6E6E6'],
            [0.001,'#FFFFCC'],
            [0.25, '#FED976'],
            [0.50, '#FEB24C'],
            [0.75, '#FD8D3C'],
            [1.0,  '#E31A1C']
        ]
        
        z = heatmap_matrix.values
        zmax = float(z.max()) if z.size else 1.0
        if zmax <= 0: zmax = 1.0
        
        fig = px.imshow(
            z,
            x=col_labels,
            y=heatmap_matrix.index,
            text_auto=True,
            aspect='auto',
            color_continuous_scale=colorscale,
            zmin=0, zmax=zmax,
            labels=dict(x="Time Interval (30 min)", y="Store Name", color="Receipts")
        )
        
        fig.update_xaxes(side='top')
        fig.update_layout(height=max(600, 25*len(heatmap_matrix.index)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Peak Active Tills")
        
        df_tills = df.copy()
        df_tills['TRN_DATE'] = pd.to_datetime(df_tills['TRN_DATE'], errors='coerce')
        df_tills = df_tills.dropna(subset=['TRN_DATE'])
        
        df_tills['TIME_INTERVAL'] = df_tills['TRN_DATE'].dt.floor('30T')
        df_tills['TIME_ONLY'] = df_tills['TIME_INTERVAL'].dt.time
        
        start_time = pd.Timestamp("00:00:00")
        intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        
        till_counts = (
            df_tills.groupby(['STORE_NAME','TIME_ONLY'])['Till_Code']
            .nunique()
            .reset_index(name='UNIQUE_TILLS')
        )
        
        heatmap = till_counts.pivot(index='STORE_NAME', columns='TIME_ONLY', values='UNIQUE_TILLS').fillna(0)
        
        for t in intervals:
            if t not in heatmap.columns:
                heatmap[t] = 0
        heatmap = heatmap[intervals]
        
        heatmap['MAX_TILLS'] = heatmap.max(axis=1).astype(int)
        heatmap = heatmap.sort_values('MAX_TILLS', ascending=False)
        
        mat = heatmap.drop(columns=['MAX_TILLS']).values
        
        colorscale = [
            [0.0,  '#E6E6E6'],
            [0.001,'#FFFFCC'],
            [0.25, '#FED976'],
            [0.50, '#FEB24C'],
            [0.75, '#FD8D3C'],
            [1.0,  '#E31A1C']
        ]
        zmax = float(mat.max()) if mat.size else 1.0
        if zmax <= 0: zmax = 1.0
        
        fig = px.imshow(
            mat,
            x=col_labels,
            y=heatmap.index,
            text_auto=True,
            aspect='auto',
            color_continuous_scale=colorscale,
            zmin=0, zmax=zmax,
            labels=dict(x="Time Interval (30 min)", y="Store Name", color="Unique Tills")
        )
        
        fig.update_xaxes(side='top')
        fig.update_layout(height=max(600, 25*len(heatmap.index)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Cashier Performance")
        
        df_cash = df.copy()
        df_cash['TRN_DATE'] = pd.to_datetime(df_cash['TRN_DATE'], errors='coerce')
        df_cash = df_cash.dropna(subset=['TRN_DATE'])
        
        if 'CUST_CODE' not in df_cash.columns:
            st.warning("CUST_CODE not available")
        else:
            receipt_duration = (
                df_cash.groupby(['STORE_NAME', 'CUST_CODE'], as_index=False)
                .agg(Start_Time=('TRN_DATE', 'min'), End_Time=('TRN_DATE', 'max'))
            )
            receipt_duration['Duration_Sec'] = (receipt_duration['End_Time'] - receipt_duration['Start_Time']).dt.total_seconds()
            receipt_duration['Duration_Sec'] = receipt_duration['Duration_Sec'].fillna(0)
            
            receipt_items = (
                df_cash.groupby(['STORE_NAME', 'CUST_CODE'], as_index=False)['ITEM_CODE']
                .nunique()
                .rename(columns={'ITEM_CODE': 'Unique_Items'})
            )
            
            receipt_stats = pd.merge(receipt_duration, receipt_items, on=['STORE_NAME', 'CUST_CODE'], how='left')
            
            store_summary = (
                receipt_stats.groupby('STORE_NAME', as_index=False)
                .agg(
                    Total_Customers=('CUST_CODE', 'nunique'),
                    Avg_Time_per_Customer_Min=('Duration_Sec', lambda s: s.mean() / 60),
                    Avg_Items_per_Receipt=('Unique_Items', 'mean')
                )
            )
            
            store_summary = store_summary.sort_values('Avg_Time_per_Customer_Min', ascending=True).reset_index(drop=True)
            
            st.dataframe(store_summary, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("Till Usage by Store")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            store_select = st.selectbox("Select Store:", sorted(df['STORE_NAME'].unique()), key="till_store")
        
        df_till = df[df['STORE_NAME'] == store_select].copy()
        df_till['TRN_DATE'] = pd.to_datetime(df_till['TRN_DATE'], errors='coerce')
        df_till = df_till.dropna(subset=['TRN_DATE'])
        
        df_till['TIME_SLOT'] = df_till['TRN_DATE'].dt.floor('30T')
        df_till['TIME_ONLY'] = df_till['TIME_SLOT'].dt.time
        
        till_activity = (
            df_till.groupby(['Till_Code', 'TIME_ONLY'], as_index=False)
            .agg(Receipts=('CUST_CODE', 'nunique'))
        )
        
        start_time = pd.Timestamp("00:00:00")
        intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
        
        pivot = (
            till_activity.pivot(index='Till_Code', columns='TIME_ONLY', values='Receipts')
            .reindex(columns=intervals)
            .fillna(0)
        )
        
        fig = px.imshow(
            pivot.values,
            x=[f"{t.hour:02d}:{t.minute:02d}" for t in intervals],
            y=pivot.index,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Blues',
            title=f"Till Activity - {store_select}"
        )
        
        fig.update_xaxes(side='top', tickangle=45)
        fig.update_layout(height=max(450, 25*len(pivot)))
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# SECTION 4: CUSTOMER TRAFFIC
# ============================================
elif section == "Customer Traffic":
    st.title("👥 Customer Traffic Analysis")
    
    tab1, tab2 = st.tabs(["Store-wise", "Department-wise"])
    
    with tab1:
        st.subheader("Store Customer Traffic Patterns")
        
        store_select = st.selectbox("Select Store:", sorted(df['STORE_NAME'].unique()), key="traffic_store")
        
        df_traffic = df[df['STORE_NAME'] == store_select].copy()
        df_traffic['TRN_DATE'] = pd.to_datetime(df_traffic['TRN_DATE'], errors='coerce')
        df_traffic = df_traffic.dropna(subset=['TRN_DATE'])
        
        df_traffic['TIME_INTERVAL'] = df_traffic['TRN_DATE'].dt.floor('30T')
        df_traffic['TIME_ONLY'] = df_traffic['TIME_INTERVAL'].dt.time
        
        start_time = pd.Timestamp("00:00:00")
        intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        
        tmp = (
            df_traffic.groupby(['DEPARTMENT','TIME_ONLY'])['CUST_CODE']
            .nunique()
            .reset_index(name='Unique_Customers')
        )
        
        pivot = tmp.pivot(index='DEPARTMENT', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
        for t in intervals:
            if t not in pivot.columns:
                pivot[t] = 0
        pivot = pivot[intervals]
        
        pivot['TOTAL'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL', ascending=False)
        y_labels = pivot.index.tolist()
        totals = pivot['TOTAL'].astype(int).tolist()
        mat = pivot.drop(columns=['TOTAL']).values
        
        colorscale = [
            [0.0,  '#E6E6E6'],
            [0.001,'#e0f3db'],
            [0.25, '#a8ddb5'],
            [0.50, '#43a2ca'],
            [0.75, '#0868ac'],
            [1.0,  '#084081']
        ]
        zmax = float(mat.max()) if mat.size else 1.0
        if zmax <= 0: zmax = 1.0
        
        fig = px.imshow(
            mat,
            x=col_labels,
            y=y_labels,
            text_auto=True,
            aspect='auto',
            color_continuous_scale=colorscale,
            zmin=0,
            zmax=zmax,
            labels=dict(x="Time of Day", y="Department", color="Customers")
        )
        
        fig.update_xaxes(side='top')
        fig.update_layout(height=max(400, 25*len(y_labels)))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Department Customer Traffic Patterns")
        
        dept_select = st.selectbox("Select Department:", sorted(df['DEPARTMENT'].unique()), key="traffic_dept")
        
        df_traffic = df[df['DEPARTMENT'] == dept_select].copy()
        df_traffic['TRN_DATE'] = pd.to_datetime(df_traffic['TRN_DATE'], errors='coerce')
        df_traffic = df_traffic.dropna(subset=['TRN_DATE'])
        
        df_traffic['TIME_INTERVAL'] = df_traffic['TRN_DATE'].dt.floor('30T')
        df_traffic['TIME_ONLY'] = df_traffic['TIME_INTERVAL'].dt.time
        
        start_time = pd.Timestamp("00:00:00")
        intervals = [(start_time + timedelta(minutes=30*i)).time() for i in range(48)]
        col_labels = [f"{t.hour:02d}:{t.minute:02d}" for t in intervals]
        
        tmp = (
            df_traffic.groupby(['STORE_NAME','TIME_ONLY'])['CUST_CODE']
            .nunique()
            .reset_index(name='Unique_Customers')
        )
        
        pivot = tmp.pivot(index='STORE_NAME', columns='TIME_ONLY', values='Unique_Customers').fillna(0)
        for t in intervals:
            if t not in pivot.columns:
                pivot[t] = 0
        pivot = pivot[intervals]
        
        pivot['TOTAL'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL', ascending=False)
        y_labels = pivot.index.tolist()
        mat = pivot.drop(columns=['TOTAL']).values
        
        colorscale = [
            [0.0,  '#E6E6E6'],
            [0.001,'#e0f3db'],
            [0.25, '#a8ddb5'],
            [0.50, '#43a2ca'],
            [0.75, '#0868ac'],
            [1.0,  '#084081']
        ]
        zmax = float(mat.max()) if mat.size else 1.0
        if zmax <= 0: zmax = 1.0
        
        fig = px.imshow(
            mat,
            x=col_labels,
            y=y_labels,
            text_auto=True,
            aspect='auto',
            color_continuous_scale=colorscale,
            zmin=0,
            zmax=zmax,
            labels=dict(x="Time of Day", y="Store", color="Customers")
        )
        
        fig.update_xaxes(side='top')
        fig.update_layout(height=max(400, 25*len(y_labels)))
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# SECTION 5: LOYALTY
# ============================================
elif section == "Loyalty":
    st.title("💳 Loyalty Analysis")
    
    tab1, tab2 = st.tabs(["Branch Overview", "Customer Details"])
    
    with tab1:
        st.subheader("Loyal Customers by Branch")
        
        dfL = df[df['LOYALTY_CUSTOMER_CODE'].notna() & 
                 (df['LOYALTY_CUSTOMER_CODE'].astype(str).str.strip() != '')].copy()
        
        if dfL.empty:
            st.warning("No loyalty data available")
        else:
            receipts = (
                dfL.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False)
                .agg(Basket_Value=('NET_SALES','sum'))
            )
            
            per_branch_multi = (
                receipts.groupby(['STORE_NAME','LOYALTY_CUSTOMER_CODE'])
                .agg(Baskets_in_Store=('CUST_CODE','nunique'), Total_Value_in_Store=('Basket_Value','sum'))
                .reset_index()
            )
            
            per_branch_multi = per_branch_multi[per_branch_multi['Baskets_in_Store'] > 1]
            
            overview = (
                per_branch_multi.groupby('STORE_NAME', as_index=False)
                .agg(
                    Loyal_Customers_Multi=('LOYALTY_CUSTOMER_CODE','nunique'),
                    Total_Baskets_of_Those=('Baskets_in_Store','sum'),
                    Total_Value_of_Those=('Total_Value_in_Store','sum')
                )
            )
            
            overview['Avg_Baskets_per_Customer'] = np.where(
                overview['Loyal_Customers_Multi'] > 0,
                overview['Total_Baskets_of_Those'] / overview['Loyal_Customers_Multi'],
                0.0
            ).round(2)
            
            overview = overview.sort_values(['Loyal_Customers_Multi','Total_Baskets_of_Those'], ascending=[False, False])
            
            overview_disp = fmt_int(overview, ['Loyal_Customers_Multi','Total_Baskets_of_Those','Total_Value_of_Those'])
            
            st.dataframe(overview_disp, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Customer Loyalty Details")
        
        dfL = df[df['LOYALTY_CUSTOMER_CODE'].notna() & 
                 (df['LOYALTY_CUSTOMER_CODE'].astype(str).str.strip() != '')].copy()
        
        if dfL.empty:
            st.warning("No loyalty data available")
        else:
            receipts = (
                dfL.groupby(['STORE_NAME','CUST_CODE','LOYALTY_CUSTOMER_CODE'], as_index=False)
                .agg(Basket_Value=('NET_SALES','sum'), First_Time=('TRN_DATE','min'))
            )
            
            global_multi_custs = (
                receipts.groupby('LOYALTY_CUSTOMER_CODE')['CUST_CODE'].nunique()
            )
            eligible = sorted(global_multi_custs[global_multi_custs > 1].index.tolist())
            
            if eligible:
                cust_code = st.selectbox("Select Customer:", eligible, key="loyalty_cust")
                
                rc = receipts[receipts['LOYALTY_CUSTOMER_CODE'] == cust_code].copy()
                
                per_store = (
                    rc.groupby('STORE_NAME', as_index=False)
                    .agg(
                        Baskets=('CUST_CODE','nunique'),
                        Total_Value=('Basket_Value','sum'),
                        First_Time=('First_Time','min'),
                        Last_Time=('First_Time','max')
                    )
                )
                
                st.subheader(f"Customer: {cust_code}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stores Visited", per_store['STORE_NAME'].nunique())
                with col2:
                    st.metric("Total Baskets", int(per_store['Baskets'].sum()))
                with col3:
                    st.metric("Total Value", f"KSh {per_store['Total_Value'].sum():,.0f}")
                
                st.dataframe(per_store, use_container_width=True, hide_index=True)
            else:
                st.info("No customers with multiple baskets found")

# ============================================
# SECTION 6: PRICING
# ============================================
elif section == "Pricing":
    st.title("💲 Pricing Analysis")
    
    tab1, tab2 = st.tabs(["Global Overview", "Store Drilldown"])
    
    with tab1:
        st.subheader("Multi-Priced SKUs Overview")
        
        dfp = df.copy()
        dfp['TRN_DATE'] = pd.to_datetime(dfp['TRN_DATE'], errors='coerce')
        dfp = dfp.dropna(subset=['TRN_DATE','STORE_NAME','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT'])
        
        dfp['SP_PRE_VAT'] = (
            dfp['SP_PRE_VAT'].astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        dfp['SP_PRE_VAT'] = pd.to_numeric(dfp['SP_PRE_VAT'], errors='coerce').fillna(0.0)
        dfp['QTY'] = pd.to_numeric(dfp['QTY'], errors='coerce').fillna(0.0)
        dfp['DATE'] = dfp['TRN_DATE'].dt.date
        dfp['PRICE2'] = dfp['SP_PRE_VAT'].map(_round2)
        
        grp = (
            dfp.groupby(['STORE_NAME','DATE','ITEM_CODE','ITEM_NAME'], as_index=False)
            .agg(
                Num_Prices=('PRICE2', lambda s: s.dropna().nunique()),
                Price_Min=('PRICE2', 'min'),
                Price_Max=('PRICE2', 'max'),
                Total_QTY=('QTY', 'sum')
            )
        )
        
        grp['Price_Spread'] = (grp['Price_Max'] - grp['Price_Min']).round(2)
        multi_price = grp[(grp['Num_Prices'] > 1) & (grp['Price_Spread'] > 0)].copy()
        multi_price['Diff_Value'] = (multi_price['Total_QTY'] * multi_price['Price_Spread']).round(2)
        
        summary = (
            multi_price.groupby('STORE_NAME', as_index=False)
            .agg(
                Items_with_MultiPrice=('ITEM_CODE','nunique'),
                Total_Diff_Value=('Diff_Value','sum'),
                Avg_Spread=('Price_Spread','mean'),
                Max_Spread=('Price_Spread','max')
            )
        )
        
        summary = summary.sort_values('Total_Diff_Value', ascending=False)
        
        fig = px.bar(
            summary.head(20).sort_values('Total_Diff_Value', ascending=True),
            x='Total_Diff_Value',
            y='STORE_NAME',
            orientation='h',
            color='Items_with_MultiPrice',
            color_continuous_scale='Blues',
            title='Top Stores by Multi-Price Impact'
        )
        
        fig.update_layout(height=max(400, 20*len(summary.head(20))))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Summary Table")
        summary_disp = fmt_int(summary, ['Items_with_MultiPrice','Total_Diff_Value','Avg_Spread','Max_Spread'])
        st.dataframe(summary_disp, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Store-level Price Analysis")
        
        store_sel = st.selectbox("Select Store:", sorted(df['STORE_NAME'].unique()), key="price_store")
        
        dfp = df[df['STORE_NAME'] == store_sel].copy()
        dfp['TRN_DATE'] = pd.to_datetime(dfp['TRN_DATE'], errors='coerce')
        dfp = dfp.dropna(subset=['TRN_DATE','ITEM_CODE','ITEM_NAME','QTY','SP_PRE_VAT'])
        
        dfp['SP_PRE_VAT'] = (
            dfp['SP_PRE_VAT'].astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        dfp['SP_PRE_VAT'] = pd.to_numeric(dfp['SP_PRE_VAT'], errors='coerce').fillna(0.0)
        dfp['QTY'] = pd.to_numeric(dfp['QTY'], errors='coerce').fillna(0.0)
        dfp['DATE'] = dfp['TRN_DATE'].dt.date
        dfp['PRICE2'] = dfp['SP_PRE_VAT'].map(_round2)
        
        per_item_day = (
            dfp.groupby(['DATE','ITEM_CODE','ITEM_NAME'], as_index=False)
            .agg(
                Num_Prices=('PRICE2', lambda s: s.dropna().nunique()),
                Price_Min=('PRICE2', 'min'),
                Price_Max=('PRICE2', 'max'),
                Total_QTY=('QTY', 'sum')
            )
        )
        
        per_item_day['Price_Spread'] = per_item_day['Price_Max'] - per_item_day['Price_Min']
        multi = per_item_day[(per_item_day['Num_Prices'] > 1) & (per_item_day['Price_Spread'] > 0)].copy()
        
        if multi.empty:
            st.info(f"✅ No multi-priced items found in {store_sel}")
        else:
            multi['Diff_Value'] = (multi['Total_QTY'] * multi['Price_Spread']).map(_round2)
            multi_sum = multi.sort_values(['DATE','Price_Spread','Total_QTY'], ascending=[False, False, False])
            
            st.subheader("Items with Multiple Prices")
            st.dataframe(multi_sum, use_container_width=True, hide_index=True)

# ============================================
# SECTION 7: REFUNDS
# ============================================
elif section == "Refunds":
    st.title("🔄 Refunds Analysis")
    
    st.subheader("Negative Receipts Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        store_filter = st.selectbox("Branch:", ['All'] + sorted(df['STORE_NAME'].unique()), key="refund_store")
    
    with col2:
        stype_filter = st.selectbox("Type:", ['All','General sales','On_account sales'], key="refund_type")
    
    with col3:
        metric_filter = st.selectbox("Rank by:", [('Value','value'),('Count','count')], key="refund_metric")
    
    with col4:
        topn_filter = st.slider("Top N:", 5, 100, 50, key="refund_topn")
    
    # Build refund data
    d = df.copy()
    d['NET_SALES'] = pd.to_numeric(d['NET_SALES'], errors='coerce').fillna(0)
    
    for c in ['STORE_NAME','CAP_CUSTOMER_CODE']:
        d[c] = d[c].astype(str).str.strip()
    
    d['Sale_Type'] = np.where(
        d['CAP_CUSTOMER_CODE'].replace({'nan':'','NaN':'','None':''})=='',
        'General sales', 'On_account sales'
    )
    
    neg = d[d['NET_SALES'] < 0].copy()
    
    if neg.empty:
        st.info("No negative receipts found")
    else:
        group_cols = ['STORE_NAME','Sale_Type','CAP_CUSTOMER_CODE']
        summary = neg.groupby(group_cols)['NET_SALES'].sum().rename('Total_Neg_Value').reset_index()
        summary['Total_Count'] = neg.groupby(group_cols).size().values if len(neg.groupby(group_cols)) > 0 else 0
        summary['Abs_Neg_Value'] = summary['Total_Neg_Value'].abs()
        
        # Apply filters
        if store_filter != 'All':
            summary = summary[summary['STORE_NAME'] == store_filter]
        
        if stype_filter != 'All':
            summary = summary[summary['Sale_Type'] == stype_filter]
        
        # Sort
        sort_col = 'Abs_Neg_Value' if metric_filter == 'value' else 'Total_Count'
        summary = summary.sort_values(sort_col, ascending=False).head(topn_filter)
        
        # Format
        summary_disp = summary.copy()
        summary_disp['Total_Neg_Value'] = summary_disp['Total_Neg_Value'].map(lambda x: f"{x:,.2f}")
        summary_disp['Total_Count'] = summary_disp['Total_Count'].map('{:,.0f}'.format)
        summary_disp['Abs_Neg_Value'] = summary_disp['Abs_Neg_Value'].map(lambda x: f"{x:,.2f}")
        
        st.dataframe(summary_disp, use_container_width=True, hide_index=True)
        
        # Chart
        if not summary.empty:
            fig = px.bar(
                summary.sort_values('Abs_Neg_Value', ascending=True),
                x='Abs_Neg_Value',
                y='STORE_NAME',
                orientation='h',
                color='Total_Count',
                color_continuous_scale='Reds',
                title='Refund Impact by Store'
            )
            
            fig.update_layout(height=max(400, 20*len(summary)))
            
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("📊 Daily Deck Analytics Dashboard\nVersion 1.0")
