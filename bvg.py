import streamlit as st
import pandas as pd
import joblib
import numpy as np
from numpy import array

# Load the trained model
mdl= joblib.load(r"C:\Users\viper\Desktop\NOTES\AI_ML_DS\Projects\Beverages_price\beverages_price.pkl")     

st.subheader("🍺 Predict Beverage Price ")

# User Inputs
# Define categorical mappings
beverage_categories = ['Water', 'Soft Drinks', 'Juices', 'Alcoholic Beverages']
customer_types = ['B2B', 'B2C']
products = [
    'Vio Wasser', 'Evian', 'Sprite', 'Rauch Multivitamin', 'Gerolsteiner', 'Sauvignon Blanc', 'Tomato Juice',
    'Vittel', 'San Pellegrino', 'Mountain Dew', 'Hohes C Orange', 'Red Bull', 'Chardonnay', 'Tanqueray',
    'Rotkäppchen Sekt', 'Mango Juice', 'Apollinaris', 'Riesling', 'Granini Apple', 'Fanta', 'Rockstar',
    'Club Mate', 'Krombacher', 'Erdinger Weißbier', 'Volvic Touch', 'Volvic', 'Schwip Schwap', 'Fritz-Kola',
    'Coca-Cola', 'Selters', 'Cranberry Juice', 'Mezzo Mix', "Beck's", 'Passion Fruit Juice', 'Monster',
    'Augustiner', 'Pepsi', 'Merlot', 'Kölsch', 'Bacardi', 'Warsteiner', 'Moët & Chandon', 'Jever',
    'Veuve Clicquot', 'Johnnie Walker', 'Havana Club', 'Jack Daniels'
]
regions = [
    'Baden-Württemberg', 'Schleswig-Holstein', 'Hamburg', 'Bayern', 'Saarland', 'Thüringen', 'Brandenburg',
    'Nordrhein-Westfalen', 'Mecklenburg-Vorpommern', 'Sachsen-Anhalt', 'Niedersachsen', 'Rheinland-Pfalz',
    'Bremen', 'Sachsen', 'Hessen', 'Berlin'
]

# Function to encode categorical values
def encode_category(value, category_list):
    return category_list.index(value)


# Input fields
# Dropdowns for categorical inputs
beverage_category = st.selectbox("Beverage Category", beverage_categories)
product = st.selectbox("Product", products)
customer_type = st.selectbox("Customer Type", customer_types)
region = st.selectbox("Region", regions)

# Numeric Inputs
unit_price = st.number_input("Unit Price", min_value=0.0, format="%.2f")
quantity = st.number_input("Quantity", min_value=1)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, format="%.2f")

# Encode categorical values
beverage_category_encoded = encode_category(beverage_category, beverage_categories)
product_encoded = encode_category(product, products)
customer_type_encoded = encode_category(customer_type, customer_types)
region_encoded = encode_category(region, regions)

# Predict button
if st.button("Predict Total Price"):
    input_data = np.array([[beverage_category_encoded, product_encoded, customer_type_encoded, region_encoded, unit_price, quantity, discount]])
    total_price = mdl.predict(input_data)[0]
    st.success(f"💰 Estimated Total Price: {total_price:.2f}") #python -m streamlit run bvg.py
