import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="BMW Cars Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "cleaned.csv"
    df = pd.read_csv('cleaned.csv',index_col=0)
    return df

df = load_data()

df = load_data()

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("new_model.pkl")

model = load_model()

st.title("🚗 BMW Cars Price Prediction")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "📈 EDA",
    "🤖 Model",
    "🚗 Predict"
])

# ==========================================================
# DATASET
# ==========================================================
with tab1:

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", df.isnull().sum().sum())

    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    st.write("Shape:", df.shape)

    st.dataframe(df.head())

    st.dataframe(df.describe().T) 

# ==========================================================
# EDA
# ==========================================================
with tab2:

    st.header("📊 Exploratory Data Analysis")

    st.markdown("""
    This section provides a detailed analysis of the used car dataset,
    including price distribution, car characteristics, brand analysis,
    fuel type, transmission, mileage, engine capacity, and relationships
    between important features and car prices.
    """)

    st.divider()

    # ======================================================
    # EDA KPIs
    # ======================================================

    st.subheader("📌 Dataset Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Price",
            f"{df['Price'].mean():,.2f}"
        )

    with col2:
        st.metric(
            "Median Price",
            f"{df['Price'].median():,.2f}"
        )

    with col3:
        st.metric(
            "Most Popular Brand",
            df["Brand_name"].value_counts().idxmax()
        )

    with col4:
        st.metric(
            "Average Car Age",
            f"{2026 - df['Year'].mean():.1f} Years"
        )

    st.divider()

    # ======================================================
    # PRICE ANALYSIS
    # ======================================================

    st.header("💰 Price Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Price",
            nbins=50,
            title="Price Distribution",
            marginal="box"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            y="Price",
            title="Price Boxplot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        f"""
        The average car price is **{df['Price'].mean():,.2f}**,
        while the median price is **{df['Price'].median():,.2f}**.

        The difference between the mean and median can indicate
        the presence of expensive cars that may affect the overall
        price distribution.
        """
    )

    st.divider()

    # ======================================================
    # PRICE BY BRAND
    # ======================================================

    st.header("🏷️ Brand Analysis")

    brand_stats = (
        df.groupby("Brand_name")["Price"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
        .head(15)
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            brand_stats,
            x="Brand_name",
            y="mean",
            title="Average Price by Brand",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        brand_count = (
            df["Brand_name"]
            .value_counts()
            .head(15)
            .reset_index()
        )

        brand_count.columns = [
            "Brand_name",
            "Count"
        ]

        fig = px.bar(
            brand_count,
            x="Brand_name",
            y="Count",
            title="Most Common Car Brands",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.dataframe(
        brand_stats,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ======================================================
    # FUEL TYPE ANALYSIS
    # ======================================================

    st.header("⛽ Fuel Type Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fuel_count = (
            df["Fuel_Type"]
            .value_counts()
            .reset_index()
        )

        fuel_count.columns = [
            "Fuel_Type",
            "Count"
        ]

        fig = px.pie(
            fuel_count,
            names="Fuel_Type",
            values="Count",
            hole=0.4,
            title="Fuel Type Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            x="Fuel_Type",
            y="Price",
            color="Fuel_Type",
            title="Price Distribution by Fuel Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # TRANSMISSION ANALYSIS
    # ======================================================

    st.header("⚙️ Transmission Analysis")

    col1, col2 = st.columns(2)

    with col1:

        transmission_count = (
            df["Transmission"]
            .value_counts()
            .reset_index()
        )

        transmission_count.columns = [
            "Transmission",
            "Count"
        ]

        fig = px.bar(
            transmission_count,
            x="Transmission",
            y="Count",
            title="Transmission Distribution",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            x="Transmission",
            y="Price",
            color="Transmission",
            title="Price by Transmission Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # YEAR ANALYSIS
    # ======================================================

    st.header("📅 Year Analysis")

    yearly = (
        df.groupby("Year")
        .agg(
            Average_Price=("Price", "mean"),
            Number_of_Cars=("Price", "count")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.line(
            yearly,
            x="Year",
            y="Average_Price",
            markers=True,
            title="Average Car Price by Year"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            yearly,
            x="Year",
            y="Number_of_Cars",
            title="Number of Cars by Year"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # MILEAGE VS PRICE
    # ======================================================

    st.header("🚗 Mileage vs Price")

    fig = px.scatter(
        df,
        x="Kilometers_Driven",
        y="Price",
        color="Fuel_Type",
        size="Power_bhp",
        hover_data=[
            "Brand_name",
            "Model_name",
            "Year"
        ],
        title="Relationship Between Mileage and Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info("""
    Generally, cars with higher kilometers driven tend to have lower
    prices. However, other factors such as brand, year, engine capacity,
    and power can also affect the final price.
    """)

    st.divider()

    # ======================================================
    # ENGINE CAPACITY VS PRICE
    # ======================================================

    st.header("🔧 Engine Capacity vs Price")

    fig = px.scatter(
        df,
        x="engine_capacity(cc)",
        y="Price",
        color="Fuel_Type",
        size="Power_bhp",
        hover_data=[
            "Brand_name",
            "Model_name"
        ],
        title="Engine Capacity vs Car Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # POWER VS PRICE
    # ======================================================

    st.header("⚡ Power vs Price")

    fig = px.scatter(
        df,
        x="Power_bhp",
        y="Price",
        color="Transmission",
        hover_data=[
            "Brand_name",
            "Model_name",
            "Year"
        ],
        title="Relationship Between Engine Power and Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # TOP MODELS
    # ======================================================

    st.header("🚘 Top Car Models")

    top_models = (
        df["Model_name"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    top_models.columns = [
        "Model_name",
        "Count"
    ]

    fig = px.bar(
        top_models,
        x="Model_name",
        y="Count",
        title="Most Common Car Models",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # OWNER TYPE
    # ======================================================

    st.header("👤 Owner Type Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names="Owner_Type",
            title="Distribution of Owner Types"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            x="Owner_Type",
            y="Price",
            color="Owner_Type",
            title="Price Distribution by Owner Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # CORRELATION ANALYSIS
    # ======================================================

    st.header("🔥 Correlation Analysis")

    numeric_cols = [
        "Year",
        "Kilometers_Driven",
        "Mileage",
        "engine_capacity(cc)",
        "Power_bhp",
        "Seats",
        "Price"
    ]

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # CORRELATION WITH PRICE
    # ======================================================

    st.subheader("📌 Features Correlated with Price")

    price_corr = (
        corr["Price"]
        .drop("Price")
        .sort_values()
        .reset_index()
    )

    price_corr.columns = [
        "Feature",
        "Correlation"
    ]

    fig = px.bar(
        price_corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Correlation of Numerical Features with Price",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # INTERACTIVE ANALYSIS
    # ======================================================

    st.header("🎛️ Interactive Data Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:

        x_axis = st.selectbox(
            "Select X Axis",
            numeric_columns
        )

    with col2:

        y_axis = st.selectbox(
            "Select Y Axis",
            numeric_columns,
            index=numeric_columns.index("Price")
            if "Price" in numeric_columns else 0
        )

    with col3:

        color_option = st.selectbox(
            "Color By",
            [
                "Fuel_Type",
                "Transmission",
                "Brand_name",
                "Owner_Type"
            ]
        )

    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=color_option,
        hover_data=[
            "Brand_name",
            "Model_name",
            "Year"
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("Shape:", df.shape)

    st.dataframe(df.head())

    st.dataframe(df.describe().T) 

# ==========================================================
# MODEL
# ==========================================================
with tab3:

    st.header("🤖 Machine Learning Model Performance")

    st.markdown("""
    ### 🏆 Best Model Selection

    Several Machine Learning models were trained and evaluated.
    After comparing their performance, the best-performing model
    was selected for the final prediction system.
    """)

    st.divider()

    # ==========================
    # Model Results
    # ==========================

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "CatBoost"
        ],

        "R² Score": [
            0.81,
            0.89,
            0.90,
            0.91,
            0.91,
            0.9137
        ]
    })

    # ==========================
    # Find Best Model
    # ==========================

    best_model = results.loc[
        results["R² Score"].idxmax()
    ]

    best_model_name = best_model["Model"]
    best_score = best_model["R² Score"]

    # ==========================
    # Best Model Card
    # ==========================

    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )

    st.markdown(
        f"""
        ### ⭐ {best_model_name} Regressor

        The **{best_model_name}** model achieved the highest R² score
        among all evaluated models and was selected as the final model
        for predicting car prices.
        """
    )

    st.divider()

    # ==========================
    # Metrics
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            best_model_name
        )

    with col2:
        st.metric(
            "R² Score",
            f"{best_score * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Models Evaluated",
            len(results)
        )

    st.divider()

    # ==========================
    # Model Comparison Table
    # ==========================

    st.subheader("📊 Model Comparison")

    results_display = results.copy()

    results_display["R² Score"] = (
        results_display["R² Score"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        results_display,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================
    # Model Comparison Chart
    # ==========================

    st.subheader("📈 Model Performance Comparison")

    fig = px.bar(
        results,
        x="Model",
        y="R² Score",
        text="R² Score",
        title="R² Score Comparison Between Models"
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="R² Score",
        xaxis_title="Machine Learning Model",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================
    # Final Model
    # ==========================

    st.subheader("🚀 Final Model")

    st.info(
        f"""
        The **{best_model_name}** model was selected as the final model
        because it achieved the best performance with an R² score of
        **{best_score * 100:.2f}%**.

        This model is used in the **Prediction** tab to estimate
        the price of a used car based on its features.
        """
    )

# ==========================================================
# PREDICTION
# ==========================================================
with tab4:

    st.header("🚗 Used Car Price Prediction")

    col1, col2 = st.columns(2)

    with col1:

        brand = st.selectbox(
            "Brand",
            sorted(df["Brand_name"].unique())
        )

        model_name = st.selectbox(
            "Model",
            sorted(df[df["Brand_name"] == brand]["Model_name"].unique())
        )

        location = st.selectbox(
            "Location",
            sorted(df["Location"].unique())
        )

        year = st.selectbox(
            "Year",
            sorted(df["Year"].unique(), reverse=True)
        )

        transmission = st.selectbox(
            "Transmission",
            sorted(df["Transmission"].unique())
        )

        fuel = st.selectbox(
            "Fuel Type",
            sorted(df["Fuel_Type"].unique())
        )

    with col2:

        owner = st.selectbox(
            "Owner Type",
            sorted(df["Owner_Type"].unique())
        )

        km = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=500000,
            value=50000
        )

        mileage = st.number_input(
            "Mileage",
            min_value=float(df["Mileage"].min()),
            max_value=float(df["Mileage"].max()),
            value=float(df["Mileage"].median())
        )

        engine = st.number_input(
            "Engine Capacity (cc)",
            min_value=int(df["engine_capacity(cc)"].min()),
            max_value=int(df["engine_capacity(cc)"].max()),
            value=int(df["engine_capacity(cc)"].median())
        )

        power = st.number_input(
            "Power (bhp)",
            min_value=float(df["Power_bhp"].min()),
            max_value=float(df["Power_bhp"].max()),
            value=float(df["Power_bhp"].median())
        )

        seats = st.selectbox(
            "Seats",
            sorted(df["Seats"].unique())
        )

    if st.button("Predict Price"):

        sample = pd.DataFrame({

            "Location":[location],
            "Year":[year],
            "Kilometers_Driven":[km],
            "Fuel_Type":[fuel],
            "Transmission":[transmission],
            "Owner_Type":[owner],
            "Mileage":[mileage],
            "engine_capacity(cc)":[engine],
            "Power_bhp":[power],
            "Seats":[seats],
            "Brand_name":[brand],
            "Model_name":[model_name]

        })

        prediction = model.predict(sample)[0]

        st.success(f"💰 Estimated Market Price: ${prediction:,.2f}")

        st.info("⚠️ This is an estimated market price based on the trained machine learning model.")
