
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="BMW Car Prices Project",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("cleaned.csv", index_col=0)

df = load_data()

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 22px;
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.section-title {
    font-size: 30px;
    font-weight: bold;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
    background-color: #f8f9fa;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# Header
# ==========================================

st.markdown(
    '<div class="main-title">🚗 BMW Cars Prices Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Project for Used BMW Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# Image
# ==========================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.image(
        "download.jpg",
        use_container_width=True
    )


# ==========================================
# Project Introduction
# ==========================================

st.markdown(
    '<div class="section-title">📌 About The Project</div>',
    unsafe_allow_html=True
)

st.write("""
This project focuses on analyzing used BMW cars and predicting their
selling prices using Machine Learning techniques.

The application provides an interactive platform where users can explore
the dataset, understand the most important factors affecting car prices,
compare different Machine Learning models, and predict the estimated price
of a BMW car based on its characteristics.
""")


st.divider()


# ==========================================
# Dataset Overview
# ==========================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Cars",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Average Price",
        f"£{df['Price'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Average Mileage",
        f"{df['Mileage'].mean():,.0f}"
    )


st.divider()

# ==========================================
# Column Descriptions
# ==========================================

st.markdown(
    '<div class="section-title">📖 Column Descriptions</div>',
    unsafe_allow_html=True
)

with st.expander("🚗 Model"):
    st.write(
        "The BMW car model, such as 1 Series, 3 Series, 5 Series, X1, X3, and X5."
    )

with st.expander("📅 Year"):
    st.write(
        "The year in which the BMW car was manufactured."
    )

with st.expander("💰 Price"):
    st.write(
        "The selling price of the BMW car in British Pounds (GBP). "
        "This is the target variable predicted by the Machine Learning model."
    )

with st.expander("⚙️ Transmission"):
    st.write(
        "The type of transmission used by the car, such as Automatic, Manual, or Semi-Auto."
    )

with st.expander("🛣️ Mileage"):
    st.write(
        "The total distance the car has been driven, measured in miles."
    )

with st.expander("⛽ Fuel Type"):
    st.write(
        "The type of fuel used by the car, such as Petrol, Diesel, Hybrid, or Electric."
    )

with st.expander("💷 Tax"):
    st.write(
        "The annual road tax associated with the vehicle, measured in British Pounds (GBP)."
    )

with st.expander("⛽ MPG"):
    st.write(
        "The fuel efficiency of the car, measured in miles per gallon."
    )

with st.expander("🔧 Engine Size"):
    st.write(
        "The engine displacement of the car, measured in litres, "
        "such as 1.5L, 2.0L, or 3.0L."
    )

st.divider()

# ==========================================
# Dataset Features
# ==========================================

st.markdown(
    '<div class="section-title">🔍 Dataset Features</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    ### 🚗 Car Information

    - **Model:** BMW car model
    - **Year:** Manufacturing year
    - **Transmission:** Automatic, Manual, Semi-Auto
    - **Fuel Type:** Petrol, Diesel, Hybrid, Electric
    """)

with col2:

    st.markdown("""
    ### 📈 Car Specifications

    - **Mileage:** Distance driven by the car
    - **Tax:** Annual road tax
    - **MPG:** Fuel efficiency
    - **Engine Size:** Engine displacement in litres
    """)


st.divider()


# ==========================================
# Project Objectives
# ==========================================

st.markdown(
    '<div class="section-title">🎯 Project Objectives</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    ### 📊 Data Analysis

    Explore the dataset and discover patterns
    and relationships between car features
    and prices.
    """)

with col2:

    st.markdown("""
    ### 🤖 Machine Learning

    Train and compare different Machine Learning
    models to find the best-performing model.
    """)

with col3:

    st.markdown("""
    ### 💰 Price Prediction

    Predict the estimated price of a used BMW car
    based on its specifications.
    """)


st.divider()


# ==========================================
# Navigation Guide
# ==========================================

st.markdown(
    '<div class="section-title">🧭 Explore The Project</div>',
    unsafe_allow_html=True
)

st.info("""
Use the sidebar to navigate through the application:

📊 **Dataset** → Explore the dataset and its features.

📈 **EDA** → Analyze the data using interactive visualizations.

🤖 **Model** → Compare Machine Learning models and their performance.

🚗 **Predict** → Enter car specifications and predict the estimated price.
""")


st.divider()


# ==========================================
# Footer
# ==========================================

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:20px;">
        <h4>BMW Cars Prices Prediction</h4>
        <p>Machine Learning & Streamlit Project</p>
    </div>
    """,
    unsafe_allow_html=True
)