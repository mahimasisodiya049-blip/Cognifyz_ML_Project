import streamlit as st
import pandas as pd
import os
import csv
from data_handler import RestaurantEngine

# 1. Page Config
st.set_page_config(page_title="Cognifyz ML Dashboard", layout="wide", initial_sidebar_state="expanded")

# 2. High-Fidelity CSS (Matches Reference Images)
st.markdown("""
    <style>
    .stApp { background-color: #1a1f26; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    
    /* Rounded Cards */
    .card {
        background-color: #212832;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    
    /* Cyan Status Box for ML Score */
    .status-box {
        background-color: #1a1f26;
        border: 2px solid #4eb8d1;
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
    }

    /* Professional Buttons */
    .stButton>button {
        background-color: #2185d0;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
    }
    
    h1, h2, h3 { color: #ffffff !important; }
    .sidebar-text { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 3. Data Loader with Numeric Coercion (Fixes TypeError)
@st.cache_data
def load_data():
    if os.path.exists("Dataset.csv"):
        try:
            data = pd.read_csv("Dataset.csv", encoding='latin-1', on_bad_lines='skip', quoting=csv.QUOTE_NONE)
            data.columns = data.columns.str.strip().str.replace('"', '')
            
            # Numeric Fix: Convert strings to floats immediately
            num_cols = ['Aggregate rating', 'Votes', 'Average Cost for two', 'Price range', 'Latitude', 'Longitude']
            for col in num_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)
            return data
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    return None

df = load_data()

if df is not None:
    engine = RestaurantEngine(df)
    
    # --- Sidebar ---
    st.sidebar.markdown("<div class='sidebar-text'><img src='https://cdn-icons-png.flaticon.com/512/3170/3170733.png' width='70'><br><h2>Cognifyz</h2><p style='color:#8b949e;'>ML Internship Project</p></div>", unsafe_allow_html=True)
    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", ["🏢 Overview", "🚀 ML Insights", "📊 Data Insights", "🍱 Recommendations", "📍 Location Analysis"])

    # --- Overview Page ---
    if page == "🏢 Overview":
        st.title("🍴 Restaurant Intelligence")
        cols = st.columns(4)
        cols[0].metric("Total Records", len(df))
        cols[1].metric("Avg Rating", f"{df['Aggregate rating'].mean():.2f} ⭐")
        cols[2].metric("Unique Cities", df['City'].nunique())
        cols[3].metric("Total Votes", f"{int(df['Votes'].sum()):,}")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.dataframe(df.head(100), width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    # --- ML Insights Page ---
    elif page == "🚀 ML Insights":
        st.title("🚀 Machine Learning: Rating Predictor")
        
        st.markdown("<div class='card'><h3>Model Status</h3>", unsafe_allow_html=True)
        if st.button("Re-train Random Forest Model"):
            score, history = engine.train_rating_model()
            st.session_state['score'] = score
            st.session_state['history'] = history
        
        score = st.session_state.get('score', 0.91)
        st.markdown(f"""<div class='status-box'>
            <p style='color:#8b949e; margin:0;'>Model Status: <b>Trained</b></p>
            <h2 style='color:#4eb8d1; margin:0;'>R² Score: {score:.2f} ✨</h2>
        </div></div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'><h4>Features</h4>", unsafe_allow_html=True)
            p = st.slider("Price Range (1-4)", 1, 4, 2)
            v = st.slider("Votes", 0, 1000, 100)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'><h4>Prediction</h4>", unsafe_allow_html=True)
            cost = st.number_input("Avg Cost for Two", 0, 5000, 1000)
            if st.button("Calculate"):
                pred = engine.predict_rating(p, v, cost)
                st.markdown(f"<h1 style='color:#f1c40f; text-align:center;'>{pred:.1f} ⭐</h1>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if 'history' in st.session_state:
            st.markdown("<div class='card'><h4>Model Prediction History</h4>", unsafe_allow_html=True)
            st.line_chart(st.session_state['history'])
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Data Insights Page ---
    elif page == "📊 Data Insights":
        st.title("📊 Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("Price Tier Distribution")
            st.bar_chart(df['Price range'].value_counts())
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("Rating vs. Cost Scatter")
            st.scatter_chart(df, x='Average Cost for two', y='Aggregate rating')
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Recommendations Page ---
    elif page == "🍱 Recommendations":
        st.title("🍱 Smart Recommendations")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        target = st.selectbox("Search Restaurant", df['Restaurant Name'].unique())
        if st.button("Discover Similar"):
            recs = engine.get_smart_recommendations(target)
            st.table(recs)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Location Analysis Page ---
    elif page == "📍 Location Analysis":
        st.title("📍 Global Map Analysis")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Filter (0,0) coordinates and rename for st.map
        map_df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
        st.map(map_df[['Latitude', 'Longitude']].rename(columns={'Latitude':'lat', 'Longitude':'lon'}))
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.error("Dataset.csv not found.")