import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Set up page configuration
st.set_page_config(
    page_title="Shopper Spectrum Dashboard",
    page_icon="🛒",
    layout="wide"
)

# --- PREMIUM MIDNIGHT EXECUTIVE DARK STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main Background - High-End Midnight Dark Slate */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(30, 41, 59, 0.7) 0, transparent 50%),
            radial-gradient(at 100% 0%, rgba(15, 23, 42, 1) 0, transparent 50%);
    }
    
    /* Pure Dark Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Text Adjustments inside Sidebar for legibility */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #f1f5f9 !important;
    }
    
    /* --- ULTIMATE METRIC KPI VISIBILITY FIX --- */
    div[data-testid="stMetricBlock"] {
        background-color: #1e293b !important;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
        border-left: 5px solid #38bdf8; /* Glowing Light Blue Left Border */
    }
    
    /* Target the KPI Label/Title directly at the root container */
    [data-testid="stMetric"] label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }
    
    /* Target the KPI Large Value directly */
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* --- STATIC TABLE VISIBILITY FIX --- */
    div[data-testid="stTable"] table {
        color: #f8fafc !important; /* Makes table cell text bright & legible */
        background-color: #1e293b !important; /* Matches card backgrounds */
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }
    
    div[data-testid="stTable"] th {
        background-color: #334155 !important; /* Slightly distinct header color */
        color: #38bdf8 !important; /* Accent light blue for column titles */
        font-weight: 600 !important;
    }

    div[data-testid="stTable"] td {
        border-bottom: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }
    
    /* Luminous Dashboard Section Headers */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    /* Clean Subtitle Captions */
    .stMarkdown p {
        color: #cbd5e1 !important;
    }
    
    /* Clean Divider Lines */
    hr {
        border-color: #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_similarity_matrix():
    try:
        with open("product_similarity.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def predict_customer_cluster(recency, frequency, monetary):
    means = np.array([90.0, 4.5, 1200.0])  
    stds = np.array([100.0, 7.0, 3000.0])  
    
    scaled_inputs = (np.array([recency, frequency, monetary]) - means) / stds
    centroids = {
        0: np.array([-0.6,  1.5,  2.1]),
        1: np.array([-0.2, -0.1, -0.2]),
        2: np.array([ 0.1, -0.4, -0.3]),
        3: np.array([ 1.4, -0.5, -0.3])
    }
    
    best_cluster = None
    min_distance = float('inf')
    for cluster_id, center_vector in centroids.items():
        distance = np.linalg.norm(scaled_inputs - center_vector)
        if distance < min_distance:
            min_distance = distance
            best_cluster = cluster_id

    cluster_mapping = {
        0: {"label": "High-Value", "desc": "Regular, frequent, recent, and big spenders."},
        1: {"label": "Regular", "desc": "Steady purchasers but not premium."},
        2: {"label": "Occasional", "desc": "Rare, occasional purchases."},
        3: {"label": "At-Risk", "desc": "Haven’t purchased in a long time."}
    }
    return cluster_mapping.get(best_cluster)

# --- NAVIGATION SIDEBAR ---
st.sidebar.header("Navigation Menu")
module_choice = st.sidebar.radio(
    "Select a Section:",
    ["🏠 1. Home & EDA Insights", "👥 2. Customer Segmentation", "🛍️ 3. Product Recommendations"]
)

# --- MODULE 1: HOME & EDA INSIGHTS ---
if module_choice == "🏠 1. Home & EDA Insights":
    st.title("🛒 Shopper Spectrum Analytics Dashboard")
    st.markdown("**Domain:** E-Commerce and Retail Analytics")
    
    # Key Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analysis Period", "2022 - 2023")
    m2.metric("Target Segments", "4 Groups")
    m3.metric("Clustering Method", "K-Means")
    m4.metric("Recommendation Style", "Item-Based Filtering")
    
    st.markdown("---")
    st.header("📊 Required Project Visual Insights (All 8 Charts)")
    
    # ROW 1
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("1. Transaction Volume by Country")
        if os.path.exists("transaction_by_country.png"):
            st.image("transaction_by_country.png", use_container_width=True)
    with row1_col2:
        st.subheader("2. Top-Selling Products")
        if os.path.exists("top_products.png"):
            st.image("top_products.png", use_container_width=True)

    st.markdown("---")

    # ROW 2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.subheader("3. Purchase Trends Over Time")
        if os.path.exists("purchase_trends.png"):
            st.image("purchase_trends.png", use_container_width=True)
    with row2_col2:
        st.subheader("4. Product Recommendation Heatmap")
        if os.path.exists("product_heatmap.png"):
            st.image("product_heatmap.png", use_container_width=True)

    st.markdown("---")

    # ROW 3
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.subheader("5. The Elbow Curve Validation")
        if os.path.exists("rfm_elbow_curve.png"):
            st.image("rfm_elbow_curve.png", use_container_width=True)
    with row3_col2:
        st.subheader("6. Customer Segment Clusters")
        if os.path.exists("customer_clusters_2d.png"):
            st.image("customer_clusters_2d.png", use_container_width=True)

    st.markdown("---")

    # ROW 4
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        st.subheader("7. Customer Monetary Distribution")
        if os.path.exists("monetary_distribution.png"):
            st.image("monetary_distribution.png", use_container_width=True)
    with row4_col2:
        st.subheader("8. RFM Feature Distributions")
        if os.path.exists("rfm_distributions.png"):
            st.image("rfm_distributions.png", use_container_width=True)

# --- MODULE 2: CUSTOMER SEGMENTATION ---
elif module_choice == "👥 2. Customer Segmentation":
    st.header("👥 Customer Segmentation Module")
    st.write("Enter values below to instantly find a shopper's business segment.")
    
    col1, col2, col3 = st.columns(3)
    with col1: recency = st.number_input("Recency (Days since last buy):", min_value=0, max_value=365, value=30)
    with col2: frequency = st.number_input("Frequency (Total orders):", min_value=1, max_value=500, value=5)
    with col3: monetary = st.number_input("Monetary Value (Total spent):", min_value=0.0, max_value=100000.0, value=150.0)
        
    if st.button("Predict Target Segment"):
        cluster_info = predict_customer_cluster(recency, frequency, monetary)
        st.success(f"Target Segment: {cluster_info['label']}")
        st.caption(f"Details: {cluster_info['desc']}")

# --- MODULE 3: PRODUCT RECOMMENDATIONS ---
elif module_choice == "🛍️ 3. Product Recommendations":
    st.header("🛍️ Product Recommendation Module")
    similarity_df = load_similarity_matrix()
    
    if similarity_df is not None:
        product_list = similarity_df.index.tolist() if hasattr(similarity_df, 'index') else []
        if product_list:
            selected_product = st.selectbox("Select a Product:", product_list)
            if st.button("Get Recommendations"):
                recommendations = similarity_df[selected_product].sort_values(ascending=False).drop(labels=[selected_product], errors='ignore').head(5)
                rec_df = pd.DataFrame({"Recommended Product": recommendations.index, "Match Score": recommendations.values})
                rec_df["Match Score"] = rec_df["Match Score"].map(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
                st.table(rec_df)
    else:
        st.warning("Please upload 'product_similarity.pkl' to run recommendations.")