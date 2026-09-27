import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import os
import json
import re

# Load model with error handling
model = None
model_error = None
class_names = []

try:
    if os.path.exists('food101_model.keras'):
        model = load_model('food101_model.keras')
        print("Food-101 model loaded successfully")
    else:
        model_error = "Model file 'food101_model.keras' not found. Please ensure the model file is in the current directory."
except Exception as e:
    model_error = f"Error loading model: {str(e)}"

# Load class names
try:
    if os.path.exists('food101_classes.json'):
        with open('food101_classes.json', 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} food categories")
    else:
        model_error = "Class names file 'food101_classes.json' not found."
except Exception as e:
    model_error = f"Error loading class names: {str(e)}"

# Comprehensive calorie database for Food-101 categories (per 100g)
calorie_db = {
    'apple_pie': '237 calories',
    'baby_back_ribs': '290 calories',
    'baklava': '340 calories',
    'beef_carpaccio': '130 calories',
    'beef_tartare': '180 calories',
    'beet_salad': '45 calories',
    'beignets': '250 calories',
    'bibimbap': '180 calories',
    'bread_pudding': '200 calories',
    'breakfast_burrito': '220 calories',
    'bruschetta': '120 calories',
    'caesar_salad': '180 calories',
    'cannoli': '230 calories',
    'caprese_salad': '150 calories',
    'carrot_cake': '280 calories',
    'ceviche': '75 calories',
    'cheesecake': '321 calories',
    'cheese_plate': '350 calories',
    'chicken_curry': '240 calories',
    'chicken_quesadilla': '280 calories',
    'chicken_wings': '260 calories',
    'chocolate_cake': '371 calories',
    'chocolate_mousse': '280 calories',
    'churros': '250 calories',
    'clam_chowder': '100 calories',
    'club_sandwich': '350 calories',
    'crab_cakes': '180 calories',
    'creme_brulee': '280 calories',
    'croque_madame': '380 calories',
    'cup_cakes': '305 calories',
    'deviled_eggs': '140 calories',
    'donuts': '269 calories',
    'dumplings': '200 calories',
    'edamame': '121 calories',
    'eggs_benedict': '290 calories',
    'escargots': '90 calories',
    'falafel': '333 calories',
    'filet_mignon': '250 calories',
    'fish_and_chips': '520 calories',
    'foie_gras': '462 calories',
    'french_fries': '312 calories',
    'french_onion_soup': '150 calories',
    'french_toast': '229 calories',
    'fried_calamari': '175 calories',
    'fried_rice': '163 calories',
    'frozen_yogurt': '90 calories',
    'garlic_bread': '200 calories',
    'gnocchi': '130 calories',
    'greek_salad': '150 calories',
    'grilled_cheese_sandwich': '350 calories',
    'grilled_salmon': '208 calories',
    'guacamole': '160 calories',
    'gyoza': '200 calories',
    'hamburger': '295 calories',
    'hot_and_sour_soup': '90 calories',
    'hot_dog': '290 calories',
    'huevos_rancheros': '300 calories',
    'hummus': '166 calories',
    'ice_cream': '207 calories',
    'lasagna': '250 calories',
    'lobster_bisque': '130 calories',
    'lobster_roll_sandwich': '380 calories',
    'macaroni_and_cheese': '320 calories',
    'macarons': '250 calories',
    'miso_soup': '40 calories',
    'mussels': '86 calories',
    'nachos': '346 calories',
    'omelette': '154 calories',
    'onion_rings': '250 calories',
    'oysters': '60 calories',
    'pad_thai': '360 calories',
    'paella': '250 calories',
    'pancakes': '227 calories',
    'panna_cotta': '250 calories',
    'peking_duck': '340 calories',
    'pho': '150 calories',
    'pizza': '266 calories',
    'pork_chop': '250 calories',
    'poutine': '500 calories',
    'prime_rib': '320 calories',
    'pulled_pork_sandwich': '350 calories',
    'ramen': '380 calories',
    'ravioli': '140 calories',
    'red_velvet_cake': '350 calories',
    'risotto': '180 calories',
    'samosa': '260 calories',
    'sashimi': '130 calories',
    'scallops': '94 calories',
    'seaweed_salad': '50 calories',
    'shrimp_and_grits': '320 calories',
    'spaghetti_bolognese': '280 calories',
    'spaghetti_carbonara': '380 calories',
    'spring_rolls': '200 calories',
    'steak': '271 calories',
    'strawberry_shortcake': '320 calories',
    'sushi': '140 calories',
    'tacos': '226 calories',
    'takoyaki': '200 calories',
    'tiramisu': '280 calories',
    'tuna_tartare': '140 calories',
    'waffles': '291 calories'
}


def fetch_calories(prediction):
    """
    Fetch calorie information from database or Google search.
    """
    try:
        # First try local database
        prediction_clean = prediction.replace('_', ' ').lower()
        for key, value in calorie_db.items():
            if key.replace('_', ' ') == prediction_clean or key.replace('_', ' ') in prediction_clean:
                return value
        
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


def processed_img(img_path):
    try:
        if model is None:
            return None, 0, None
            
        # Food-101 model expects 128x128 input
        img = load_img(img_path, target_size=(128, 128, 3))
        img = img_to_array(img)
        # Model includes Rescaling(1./255) internally, so no need to normalize
        img = np.expand_dims(img, [0])
        
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = int(y_class)
        
        if y < len(class_names):
            res = class_names[y]
            print(res)
            
            # Get confidence score
            confidence = float(answer[0][y])
            
            # Get probability array for all classes
            probabilities = answer[0]
            
            return res, confidence, probabilities
        else:
            return None, 0, None
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, 0, None


def run():
    st.title("🍽️ Universal Food Classification")
    st.markdown("Upload any food image to classify it into 101 food categories!")
    
    # Display model error if any
    if model_error:
        st.error(model_error)
    
    img_file = st.file_uploader("Choose a Food Image", type=["jpg", "png", "jpeg"])
    
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if model is None:
            st.error("Cannot process image: Model not loaded.")
        else:
            result, confidence, probabilities = processed_img(save_image_path)
            print(result)
            
            if result is not None:
                # Format the result for display
                result_formatted = result.replace('_', ' ').title()
                
                st.success(f"**Predicted : {result_formatted}**")
                st.info(f"**Prediction Confidence : {confidence * 100:.2f}%**")
                
                # Add Top-5 predictions graph
                st.subheader("📊 Model Prediction Scores")
                
                if probabilities is not None:
                    # Get top 5 predictions
                    top_5_indices = probabilities.argsort()[-5:][::-1]
                    top_5_classes = [class_names[i].replace('_', ' ').title() for i in top_5_indices]
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
                - **Category**: {result_formatted}
                - **Confidence**: {confidence * 100:.2f}%
                - **Model**: Food-101 (101 categories)
                - **Calories**: {cal if cal else 'Not available'} (per 100g)
                """)


if __name__ == "__main__":
    run()