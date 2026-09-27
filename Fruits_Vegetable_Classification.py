import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
from tf_keras.preprocessing.image import load_img, img_to_array
from tf_keras.models import load_model
import requests
from bs4 import BeautifulSoup
import os
import re

# Load model with error handling
model = None
model_error = None

try:
    if os.path.exists('FV.h5'):
        model = load_model('FV.h5')
    else:
        model_error = "Model file 'FV.h5' not found. Please ensure the model file is in the current directory."
except Exception as e:
    model_error = f"Error loading model: {str(e)}"
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Garlic', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def fetch_calories(prediction):
    """
    Dynamically fetch calorie information from Google search with multiple fallback methods.
    """
    try:
        # Method 1: Try Google search with user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        search_query = f"calories in {prediction} 100g"
        url = f'https://www.google.com/search?q={search_query}'
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple possible selectors for calorie information
        selectors = [
            "div.BNeawe.iBp4i.AP7Wnd",
            "div.BNeawe.iBp4i",
            "div.VwiC3b",
            "span.hgKElc",
            "div.Z0LcW",
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                # Look for calorie-related text
                for element in elements:
                    text = element.get_text()
                    if any(keyword in text.lower() for keyword in ['calorie', 'kcal', 'cal']):
                        # Clean up the text
                        calories = text.strip()
                        if calories:
                            return calories
        
        # Method 2: Try to extract from knowledge panel
        knowledge_selectors = [
            "div.kno-rdesc",
            "div.xpdopen",
            "div.UDZeY",
        ]
        
        for selector in knowledge_selectors:
            elements = soup.select(selector)
            if elements:
                text = elements[0].get_text()
                if 'calorie' in text.lower():
                    # Extract calorie information
                    import re
                    calorie_match = re.search(r'(\d+\.?\d*)\s*(?:calories|kcal|cal)', text, re.IGNORECASE)
                    if calorie_match:
                        return f"{calorie_match.group(1)} calories"
        
        # Method 3: Fallback to a simple nutritional database lookup
        # This is a basic nutritional database for common fruits/vegetables
        calorie_db = {
            'apple': '52 calories',
            'banana': '89 calories',
            'beetroot': '43 calories',
            'bell pepper': '31 calories',
            'cabbage': '25 calories',
            'capsicum': '40 calories',
            'carrot': '41 calories',
            'cauliflower': '25 calories',
            'chilli pepper': '40 calories',
            'corn': '86 calories',
            'cucumber': '16 calories',
            'eggplant': '25 calories',
            'garlic': '149 calories',
            'ginger': '80 calories',
            'grapes': '69 calories',
            'jalepeno': '37 calories',
            'kiwi': '61 calories',
            'lemon': '29 calories',
            'lettuce': '15 calories',
            'mango': '60 calories',
            'onion': '40 calories',
            'orange': '47 calories',
            'paprika': '282 calories',
            'pear': '57 calories',
            'peas': '81 calories',
            'pineapple': '50 calories',
            'pomegranate': '83 calories',
            'potato': '77 calories',
            'raddish': '16 calories',
            'soy beans': '173 calories',
            'spinach': '23 calories',
            'sweetcorn': '86 calories',
            'sweetpotato': '86 calories',
            'tomato': '18 calories',
            'turnip': '28 calories',
            'watermelon': '30 calories'
        }
        
        # Try to match with database (case-insensitive)
        prediction_lower = prediction.lower()
        for key, value in calorie_db.items():
            if key in prediction_lower or prediction_lower in key:
                return value
        
        return None
        
    except Exception as e:
        st.warning("Could not fetch calorie information from Google")
        print(f"Calorie fetch error: {e}")
        return None


def processed_img(img_path):
    try:
        if model is None:
            return None, 0, None
            
        img = load_img(img_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        res = labels[y]
        print(res)
        
        # Get confidence score
        confidence = float(answer[0][y])
        
        # Get probability array for all classes
        probabilities = answer[0]
        
        return res.capitalize(), confidence, probabilities
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, 0, None


def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    
    # Display model error if any
    if model_error:
        st.error(model_error)
    
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            if model is None:
                st.error("Cannot process image: Model not loaded.")
            else:
                result, confidence, probabilities = processed_img(save_image_path)
                print(result)
                
                if result is not None:
                    # Fix capitalization issue - convert both to lower case for comparison
                    result_lower = result.lower()
                    vegetables_lower = [v.lower() for v in vegetables]
                    
                    if result_lower in vegetables_lower:
                        st.info('**Category : Vegetables**')
                    else:
                        st.info('**Category : Fruit**')
                    
                    st.success("**Predicted : " + result + '**')
                    st.info(f"**Prediction Confidence : {confidence * 100:.2f}%**")
                    
                    # Add Top-5 predictions graph
                    st.subheader("📊 Model Prediction Scores")
                    
                    if probabilities is not None:
                        # Get top 5 predictions
                        top_5_indices = probabilities.argsort()[-5:][::-1]
                        top_5_classes = [labels[i].capitalize() for i in top_5_indices]
                        top_5_probs = [probabilities[i] * 100 for i in top_5_indices]
                        
                        # Create DataFrame for display
                        df = pd.DataFrame({
                            'Class': top_5_classes,
                            'Confidence (%)': top_5_probs
                        })
                        
                        # Display bar chart
                        st.bar_chart(df.set_index('Class'))
                        
                        # Display table
                        st.table(df)
                    
                    cal = fetch_calories(result)
                    if cal:
                        st.warning('**' + cal + '(100 grams)**')


run()
