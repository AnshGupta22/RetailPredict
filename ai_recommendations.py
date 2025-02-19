import streamlit as st
import pandas as pd

def show_recommendations(new_data, selected_model_name):
    st.title("🤖 AI-Powered Sales Insights & Recommendations")


    # Initialize categorized recommendations with subcategories
    recommendations = []

    # Iterate through the dataset to generate recommendations
    for _, row in new_data.iterrows():
        product_id = row["product_id"]
        store_id = row["store_id"]
        predicted_sales = row[f"predicted_units_sold_{selected_model_name}"]
        inventory = row["inventory_level"]
        discount = row["discount"]
        competitor_price = row["competitor_pricing"]
        price = row["price"]
        seasonality = row["Seasonality"]

        # Restocking Alerts
        if predicted_sales > inventory:
            if inventory == 0:
                recommendations.append({
                    "Store ID": store_id,
                    "Product ID": product_id,
                    "Category": "⚠️ Restocking Alerts",
                    "Subcategory": "Urgent Restock",
                    "Recommendation": "🚨 Urgent restock needed! Out of stock."
                })
            else:
                recommendations.append({
                    "Store ID": store_id,
                    "Product ID": product_id,
                    "Category": "⚠️ Restocking Alerts",
                    "Subcategory": "Low Stock",
                    "Recommendation": "📉 Low stock alert! Predicted demand exceeds inventory."
                })

        # Discount Suggestions
        if discount < 10 and predicted_sales < row["units_sold"]:
            recommendations.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": "💡 Discount Suggestions",
                "Subcategory": "Increase Discount",
                "Recommendation": "🔻 Consider increasing discount to boost sales."
            })
        if seasonality in ["Winter", "Summer"] and discount == 0:
            recommendations.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": "💡 Discount Suggestions",
                "Subcategory": "Seasonal Discount",
                "Recommendation": f"❄️ Seasonal opportunity! Add a discount for {seasonality}."
            })

        # Price Optimization
        if price > competitor_price * 1.1:
            recommendations.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": "🔍 Price Optimization",
                "Subcategory": "Competitor Price Lower",
                "Recommendation": "📉 Competitor price is lower. Consider adjusting."
            })
        elif price < competitor_price * 0.9:
            recommendations.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": "🔍 Price Optimization",
                "Subcategory": "Consider Price Increase",
                "Recommendation": "📈 Underpriced product. Consider a price increase."
            })

        # Seasonal Insights
        if seasonality in ["Winter", "Summer"] and predicted_sales > row["units_sold"]:
            recommendations.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": "📆 Seasonal Insights",
                "Subcategory": "High Seasonal Demand",
                "Recommendation": f"🔥 High demand in {seasonality}. Ensure enough stock."
            })
        elif seasonality in ["Spring", "Autumn"] and predicted_sales < row["units_sold"]:
            recommendations.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": "📆 Seasonal Insights",
                "Subcategory": "Low Seasonal Demand",
                "Recommendation": f"🍂 Demand is dropping in {seasonality}. Adjust stock levels."
            })

    # Convert recommendations to DataFrame
    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        st.warning("No AI recommendations available for the current data.")
        return

    # 📌 Filter by Recommendation Type
    categories = ["All"] + list(recommendations_df["Category"].unique())
    selected_category = st.sidebar.selectbox("Filter by Recommendation Type", categories)

    # 📌 Display Recommendations with Expanders and Excel-like Table
    for category in recommendations_df["Category"].unique():
        if selected_category == "All" or selected_category == category:
            st.subheader(category)
            subcategories = recommendations_df[recommendations_df["Category"] == category]["Subcategory"].unique()
            
            for subcategory in subcategories:
                filtered_data = recommendations_df[
                    (recommendations_df["Category"] == category) & (recommendations_df["Subcategory"] == subcategory)
                ]
                with st.expander(f"📌 {subcategory} ({len(filtered_data)})", expanded=False):
                    st.dataframe(
                        filtered_data[["Store ID", "Product ID", "Recommendation"]],
                        hide_index=True,
                        use_container_width=True
                    )

    # 📥 Download Button for All Recommendations
    csv = recommendations_df.to_csv(index=False)
    st.download_button(
        label="📥 Download AI Recommendations",
        data=csv,
        file_name="ai_recommendations.csv",
        mime="text/csv"
    )
