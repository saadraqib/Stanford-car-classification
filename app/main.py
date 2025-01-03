import sys
sys.path.append(".")
import streamlit as st
import os
from PIL import Image
import numpy as np
import torch 
from model.main import predict_car_model
from model.classes import model_classes


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Title
st.title("196 🚗🚓🚙 Models Classifier")
st.write("This Classifier was trained on the Stanford Cars Dataset and can Classifiy 196 model of cars, Please upload your file and click Classify to predict the car's model.")



# Sidebar Section for File Upload
st.sidebar.header("🚘Upload an image of Car model")
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

# Main Section
col1, col2 = st.columns([2, 1])  # Divide layout into two columns

with col1:
    st.header("📷 Display Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("Please upload an image to display.")

with col2:
    if st.button("Classify"):
        if uploaded_file:
            # Placeholder for actual classification logic
            car = predict_car_model(image)
            st.success(f"🎯 The Car classified as: **{car}**")
        else:
            st.warning("⚠️ Please upload an image first!")

st.divider() 

st.header("🔍 Search among the car models")
search_query = st.text_input(label="", placeholder="Type a Car Model...")

# Filter and Display Matching Objects
if search_query:
    filtered_objects = [obj for obj in model_classes if search_query.lower() in obj.lower()]
    if filtered_objects:
        st.write("### Matching car model:")
        for obj in filtered_objects:
            st.write(f"- {obj}")
    else:
        st.write("Sorry, The Model can't classify this car model")