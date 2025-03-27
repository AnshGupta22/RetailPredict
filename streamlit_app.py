from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import ai_recommendations

# Load All Trained Models
models = {
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl"),
    "LightGBM": joblib.load("lightgbm_model.pkl")
}

# Set up page navigation in session state
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# Sidebar Navigation
st.sidebar.header("🔎 Navigation")
if st.sidebar.button("Go to Dashboard"):
    st.session_state.page = "Dashboard"

if st.sidebar.button("Go to AI Recommendations"):
    st.session_state.page = "ai_recommendations"
# App title
st.title("🛍️ RetailPredict: AI-Powered Sales & Inventory Optimization")
st.markdown("**Empowering retailers with data-driven sales forecasting and inventory management.**")


# File Upload for New Data
uploaded_file = st.file_uploader("Upload new sales data (CSV)", type=["csv"])
# If user tries to go on any other page display a message to uplaod a file.
# if uploaded_file is None:
#     st.warning("⚠️ Please upload a sales data file first before proceeding.")
#     st.stop()  # Stop further execution until a file is uploaded
if uploaded_file:
    # Load new data
    new_data = pd.read_csv(uploaded_file)
    if new_data.isnull().sum().sum() > 0:
        st.warning("Missing values detected! Filling missing data with median values.")
        new_data.fillna(new_data.median(), inplace=True)
    
    # Ensure required columns exist before feature engineering
    required_columns = ['inventory_level', 'price', 'discount', 'competitor_pricing', 'Seasonality']
    
    missing_cols = [col for col in required_columns if col not in new_data.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}. Please upload a valid file.")
    else:
        # Apply Feature Engineering
        new_data['demand_fluctuation'] = new_data['units_sold'] - new_data['units_ordered']
        new_data['price_discount_interaction'] = new_data['price'] * new_data['discount']
        new_data['log_price'] = np.log1p(new_data['price'])
        new_data['log_inventory'] = np.log1p(new_data['inventory_level'])
        new_data['inventory_season_interaction'] = new_data['inventory_level'] * new_data['Seasonality'].astype('category').cat.codes
        new_data['season_encoded'] = new_data['Seasonality'].astype('category').cat.codes

        # Select features used in training
        features = ['demand_fluctuation', 'inventory_level', 'log_inventory',
                    'inventory_season_interaction', 'log_price', 'price',
                    'discount', 'competitor_pricing', 'season_encoded',
                    'price_discount_interaction']
        
        X_new = new_data[features]  
        st.sidebar.header("🔧 Dashboard Controls")
        st.session_state.selected_model_name = st.sidebar.selectbox("Select Model for Prediction", list(models.keys()))
        selected_model = models[st.session_state.selected_model_name]
        if st.session_state.page == "Dashboard":
            
            # Add optional filters
            selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + list(new_data["category"].unique()))
            selected_store = st.sidebar.selectbox("Filter by Store", ["All"] + list(new_data["store_id"].unique()))

            # Apply filters
            if selected_category != "All":
                new_data = new_data[new_data["category"] == selected_category]

            if selected_store != "All":
                new_data = new_data[new_data["store_id"] == selected_store]

            # Recreate `X_new` after filtering so that row counts match and avoid errors.
            X_new = new_data[features]
            
            # Predict Sales
            new_data[f"predicted_units_sold_{st.session_state.selected_model_name}"] = selected_model.predict(X_new)

            # Store processed data in session state
            st.session_state.new_data = new_data
            
            # Display Predictions
            st.subheader(f"Predictions using {st.session_state.selected_model_name}")
            st.dataframe(new_data.head())
            # Ensuring all models predict before plotting
            for model_name, model in models.items():
                prediction_column = f"predicted_units_sold_{model_name}"

                if prediction_column not in new_data.columns:
                    new_data[prediction_column] = model.predict(X_new)
            # 📊 Sales Trend by Category (Left) & Store-Wise Sales Performance (Right)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Sales Trend by Category")
                fig = px.line(new_data, x="date", y=f"predicted_units_sold_{st.session_state.selected_model_name}",
                            color="category", title="Predicted Sales Trend by Category")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("🏬 Store-Wise Sales Performance")
                fig = px.bar(new_data.groupby("store_id")["predicted_units_sold_" + st.session_state.selected_model_name].sum().reset_index(),
                            x="store_id", y="predicted_units_sold_" + st.session_state.selected_model_name, 
                            title="Predicted Sales by Store", color="store_id")
                st.plotly_chart(fig, use_container_width=True)
            # Regional Sales Scatter Map with Better Styling
            st.subheader("🌍 Sales Distribution by Region")
            # Define approximate coordinates for each region
            region_coords = {
                "North": {"lat": 28.6139, "lon": 77.2090},  # Delhi
                "South": {"lat": 12.9716, "lon": 77.5946},  # Bangalore
                "East": {"lat": 22.5726, "lon": 88.3639},   # Kolkata
                "West": {"lat": 19.0760, "lon": 72.8777}    # Mumbai
            }
            # Assign lat/lon to the dataset
            new_data["lat"] = new_data["region"].map(lambda x: region_coords.get(x, {}).get("lat", None))
            new_data["lon"] = new_data["region"].map(lambda x: region_coords.get(x, {}).get("lon", None))
            # Drop rows without coordinates
            map_data = new_data.dropna(subset=["lat", "lon"])
            # Ensure predicted sales used for size are non-negative
            map_data[f"predicted_units_sold_{st.session_state.selected_model_name}"] = map_data[f"predicted_units_sold_{st.session_state.selected_model_name}"].clip(lower=0.1)
            # Create the scatter geo map with improved aesthetics
            fig = px.scatter_geo(map_data, 
                        lat="lat", lon="lon", 
                        size=f"predicted_units_sold_{st.session_state.selected_model_name}", 
                        hover_name="region", 
                        hover_data={f"predicted_units_sold_{st.session_state.selected_model_name}": ":.2f"},  # Shows exact sales in tooltip
                        color=f"predicted_units_sold_{st.session_state.selected_model_name}",  
                        color_continuous_scale="plasma",  
                        projection="natural earth",
                        title="Sales Distribution Across Regions")
            # Update layout for a better look
            fig.update_layout(
                geo=dict(
                    showland=True, landcolor="rgb(243, 243, 243)",  # Light grey land color
                    showocean=True, oceancolor="rgb(204, 230, 255)",  # Soft blue ocean color
                    lakecolor="rgb(204, 230, 255)",  # Consistent water color
                    showcountries=True, countrycolor="rgb(150, 150, 150)"  # Subtle country borders
                ),
                margin=dict(l=10, r=10, t=50, b=10),  # Reduce excess spacing
            )
            st.plotly_chart(fig, use_container_width=True)
            # Inventory vs Demand (Left) & Discount Impact (Right)
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("📦 Inventory vs Demand Balance")
                fig = px.scatter(new_data, x="inventory_level", y=f"predicted_units_sold_{st.session_state.selected_model_name}",
                                title="Inventory Levels vs Predicted Demand", color="store_id")
                st.plotly_chart(fig, use_container_width=True)
            with col4:
                st.subheader("💰 Discount Impact on Sales")
                fig = px.scatter(new_data, x="discount", y=f"predicted_units_sold_{st.session_state.selected_model_name}",
                                title="Effect of Discounts on Sales", color="category")
                st.plotly_chart(fig, use_container_width=True)
            # Seasonality Impact (Full Width)
            st.subheader("🌦️ Seasonality Impact on Sales")
            fig = px.box(new_data, x="Seasonality", y=f"predicted_units_sold_{st.session_state.selected_model_name}",
                        title="Predicted Sales Across Seasons", color="Seasonality")
            st.plotly_chart(fig, use_container_width=True)
            # Line Chart - Compare Model Predictions
            st.subheader("📈 Model Prediction Comparison Over Time")

            # Filter only the models that have predictions
            available_predictions = [f"predicted_units_sold_{name}" for name in models.keys() if f"predicted_units_sold_{name}" in new_data.columns]

            if len(available_predictions) > 0:
                fig = px.line(new_data, x=new_data.index, y=available_predictions,
                            labels={"value": "Predicted Sales", "index": "Time"},
                            title="Comparison of Predictions Across Models")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No predictions available for comparison. Please ensure models have been used for prediction.")
            # Actual vs Predicted Sales (Selected Model)
            if "units_sold" in new_data.columns:
                st.subheader(f"Actual vs Predicted Sales - {st.session_state.selected_model_name}")
                fig = px.bar(new_data, x=new_data.index, y=["units_sold", f"predicted_units_sold_{st.session_state.selected_model_name}"],
                            barmode='group', title="Actual vs Predicted Sales")
                st.plotly_chart(fig, use_container_width=True)
            # Model Performance Comparison
            st.subheader("Model Performance Comparison")
            # Store performance metrics for each model
            performance_data = []
            for model_name, model in models.items():
                y_pred = model.predict(X_new)

                r2 = r2_score(new_data["units_sold"], y_pred) if "units_sold" in new_data else None
                mse = mean_squared_error(new_data["units_sold"], y_pred) if "units_sold" in new_data else None
                rmse = np.sqrt(mse) if mse else None
                mae = mean_absolute_error(new_data["units_sold"], y_pred) if "units_sold" in new_data else None

                performance_data.append({
                    "Model": model_name,
                    "R² Score": r2*100 if r2 else "N/A",
                    "MSE": mse if mse else "N/A",
                    "RMSE": rmse if rmse else "N/A",
                    "MAE": mae if mae else "N/A"
                })
                performance_df = pd.DataFrame(performance_data)
            # Plot Model Performance
            fig = px.bar(performance_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
                        x="Metric", y="Score", color="Model", barmode="group",
                        title="Model Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)
            # 📥 Download Predictions
            st.download_button("📥 Download Predictions", new_data.to_csv(index=False), "predicted_sales.csv", "text/csv")

        elif st.session_state.page == "ai_recommendations":
            if "new_data" in st.session_state:
                new_data = st.session_state.new_data  # Retrieve data with predictions
                ai_recommendations.show_recommendations(new_data, st.session_state.selected_model_name)
            elif "selected_model_name" not in st.session_state:
                st.error("No model selected. Please go to the Dashboard first.")
            else:
                st.error("No data available. Please visit the Dashboard first.")
