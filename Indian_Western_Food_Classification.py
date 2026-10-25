import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import json
import re
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

# Load model and processor
model = None
processor = None
model_error = None

# Food labels for the 34 classes
food_labels = {
    "0": "Baked Potato", "1": "Crispy Chicken", "2": "Donut", "3": "Fries", 
    "4": "Hot Dog", "5": "Sandwich", "6": "Taco", "7": "Taquito", "8": "Apple Pie", 
    "9": "Burger", "10": "Butter Naan", "11": "Chai", "12": "Chapati", "13": "Cheesecake", 
    "14": "Chicken Curry", "15": "Chole Bhature", "16": "Dal Makhani", "17": "Dhokla", 
    "18": "Fried Rice", "19": "Ice Cream", "20": "Idli", "21": "Jalebi", "22": "Kaathi Rolls", 
    "23": "Kadai Paneer", "24": "Kulfi", "25": "Masala Dosa", "26": "Momos", "27": "Omelette", 
    "28": "Paani Puri", "29": "Pakode", "30": "Pav Bhaji", "31": "Pizza", "32": "Samosa", 
    "33": "Sushi"
}

# Comprehensive calorie database for 34 food categories (per 100g)
calorie_db = {
    'Baked Potato': '93 calories',
    'Crispy Chicken': '250 calories',
    'Donut': '269 calories',
    'Fries': '312 calories',
    'Hot Dog': '290 calories',
    'Sandwich': '350 calories',
    'Taco': '226 calories',
    'Taquito': '200 calories',
    'Apple Pie': '237 calories',
    'Burger': '295 calories',
    'Butter Naan': '300 calories',
    'Chai': '100 calories',
    'Chapati': '104 calories',
    'Cheesecake': '321 calories',
    'Chicken Curry': '240 calories',
    'Chole Bhature': '300 calories',
    'Dal Makhani': '180 calories',
    'Dhokla': '180 calories',
    'Fried Rice': '163 calories',
    'Ice Cream': '207 calories',
    'Idli': '60 calories',
    'Jalebi': '300 calories',
    'Kaathi Rolls': '280 calories',
    'Kadai Paneer': '200 calories',
    'Kulfi': '280 calories',
    'Masala Dosa': '180 calories',
    'Momos': '200 calories',
    'Omelette': '154 calories',
    'Paani Puri': '150 calories',
    'Pakode': '200 calories',
    'Pav Bhaji': '250 calories',
    'Pizza': '266 calories',
    'Samosa': '260 calories',
    'Sushi': '140 calories'
}

try:
    model_name = "prithivMLmods/Indian-Western-Food-34"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipForImageClassification.from_pretrained(model_name)
    print("Indian-Western-Food-34 model loaded successfully")
except Exception as e:
    model_error = f"Error loading model: {str(e)}"
    print(f"Model loading error: {e}")


def fetch_calories(prediction):
    """
    Fetch calorie information from database or Google search.
    """
    try:
        # First try local database
        if prediction in calorie_db:
            return calorie_db[prediction]
        
        # If not in database, try Google search as fallback
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        search_query = f"calories in {prediction} 100g"
        url = f'https://www.google.com/search?q={search_query}'
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple selectors
        selectors = [
            "div.BNeawe.iBp4i.AP7Wnd",
            "div.BNeawe.iBp4i",
            "div.VwiC3b",
            "span.hgKElc",
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text()
                    if any(keyword in text.lower() for keyword in ['calorie', 'kcal', 'cal']):
                        calories = text.strip()
                        if calories:
                            return calories
        
        return None
        
    except Exception as e:
        st.warning("Could not fetch calorie information")
        print(f"Calorie fetch error: {e}")
        return None


def processed_img(image):
    try:
        if model is None or processor is None:
            return None, 0, None
        
        # Convert PIL image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        
        # Get top prediction
        top_class_idx = probs.index(max(probs))
        predicted_class = food_labels[str(top_class_idx)]
        confidence = max(probs)
        
        return predicted_class, confidence, probs
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, 0, None


def run():
    st.title("🍛 Indian & Western Food Classification")
    st.markdown("Upload any food image to classify it into 34 food categories including Indian dishes like Pav Bhaji, Chapati, Samosa, etc.")
    
    # Display model error if any
    if model_error:
        st.error(model_error)
    
    img_file = st.file_uploader("Choose a Food Image", type=["jpg", "png", "jpeg"])
    
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        
        if model is None:
            st.error("Cannot process image: Model not loaded.")
        else:
            result, confidence, probabilities = processed_img(img)
            
            if result is not None:
                st.success(f"**Predicted : {result}**")
                st.info(f"**Prediction Confidence : {confidence * 100:.2f}%**")
                
                # Add Top-5 predictions graph
                st.subheader("📊 Model Prediction Scores")
                
                if probabilities is not None:
                    # Get top 5 predictions
                    top_5_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:5]
                    top_5_classes = [food_labels[str(i)] for i in top_5_indices]
                    top_5_probs = [probabilities[i] * 100 for i in top_5_indices]
                    
                    # Create DataFrame for display
                    df = pd.DataFrame({
                        'Food Item': top_5_classes,
                        'Confidence (%)': top_5_probs
                    })
                    
                    # Display bar chart
                    st.bar_chart(df.set_index('Food Item'))
                    
                    # Display table
                    st.table(df)
                
                # Fetch and display calories
                cal = fetch_calories(result)
                if cal:
                    st.warning(f'**🔥 Calories : {cal} (per 100g)**')
                else:
                    st.info("Calorie information not available for this item")
                
                # Additional food information
                st.subheader("🍴 Food Information")
                st.markdown(f"""
                - **Category**: {result}
                - **Confidence**: {confidence * 100:.2f}%
                - **Model**: Indian-Western-Food-34 (34 categories)
                - **Calories**: {cal if cal else 'Not available'} (per 100g)
                """)
                
                # Show all available categories
                with st.expander("📋 View All 34 Food Categories"):
                    indian_foods = [food_labels[str(i)] for i in range(10, 34)]
                    western_foods = [food_labels[str(i)] for i in range(10)]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🇮🇳 Indian Foods:**")
                        for food in indian_foods:
                            st.write(f"- {food}")
                    
                    with col2:
                        st.markdown("**🌍 Western Foods:**")
                        for food in western_foods:
                            st.write(f"- {food}")


if __name__ == "__main__":
    run()