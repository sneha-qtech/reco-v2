# 🍽️ Universal Food Classification System

## Overview
This is a complete food classification system that can identify **101 different food categories** including:
- Rice dishes (fried rice, risotto, paella)
- Chicken dishes (chicken curry, chicken wings, etc.)
- Egg dishes (omelette, deviled eggs, eggs benedict)
- Many other prepared foods and international cuisines

## Files Included
- `Food_Classification.py` - Main food classification application
- `food101_model.keras` - Pre-trained Food-101 model (101 categories)
- `food101_classes.json` - Class labels for 101 food categories
- `Fruits_Vegetable_Classification.py` - Original fruit/vegetable classifier (36 categories)

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Required Files Are Present
Make sure these files exist in the project directory:
- `food101_model.keras` (automatically downloaded)
- `food101_classes.json` (automatically downloaded)
- `upload_images/` directory

### 3. Run the Food Classification App
```bash
streamlit run Food_Classification.py
```

## Food Categories (101 Total)

The model can classify these food types:
- **Desserts**: apple_pie, cheesecake, chocolate_cake, ice_cream, tiramisu, etc.
- **Main Courses**: chicken_curry, steak, fish_and_chips, pizza, hamburger, etc.
- **Rice & Pasta**: fried_rice, risotto, spaghetti_bolognese, ramen, pad_thai, etc.
- **Egg Dishes**: omelette, deviled_eggs, eggs_benedict, huevos_rancheros
- **Appetizers**: guacamole, hummus, bruschetta, spring_rolls, etc.
- **Soups**: clam_chowder, french_onion_soup, miso_soup, pho, etc.
- **International**: bibimbap, paella, samosa, sushi, tacos, etc.

## Features

### ✅ Universal Food Classification
- Classifies 101 different food categories
- Works with any food image (JPG, PNG, JPEG)
- Real-time prediction with confidence scores

### ✅ Prediction Confidence
- Shows confidence percentage for each prediction
- Displays top 5 most likely food categories
- Visual bar chart of prediction probabilities

### ✅ Calorie Information
- Built-in calorie database for all 101 food categories
- Shows calories per 100g
- Fallback to Google search if not in database

### ✅ Professional UI
- Clean, modern interface
- Easy image upload
- Detailed prediction results
- Visual graphs and tables

## Comparison: Original vs New

| Feature | Original (FV.h5) | New (Food-101) |
|---------|------------------|----------------|
| Categories | 36 (fruits/vegetables) | 101 (all foods) |
| Model Size | ~50MB | ~100MB |
| Input Size | 224x224 | 128x128 |
| Foods Covered | Only raw fruits/vegetables | All food types including cooked dishes |
| Calorie Data | 36 items | 101 items |
| Confidence Display | ✅ | ✅ |
| Top-5 Graph | ✅ | ✅ |

## Usage Examples

### Classify a Chicken Dish
1. Upload image of chicken curry
2. System predicts: "Chicken Curry" with 94.5% confidence
3. Shows top 5 predictions and calories (240 cal/100g)

### Classify a Rice Dish
1. Upload image of fried rice
2. System predicts: "Fried Rice" with 87.2% confidence
3. Shows calories (163 cal/100g)

### Classify an Egg Dish
1. Upload image of omelette
2. System predicts: "Omelette" with 91.8% confidence
3. Shows calories (154 cal/100g)

## Running Both Applications

You can run both applications side by side:

### For Fruits/Vegetables (Original):
```bash
streamlit run Fruits_Vegetable_Classification.py
```

### For Universal Food Classification (New):
```bash
streamlit run Food_Classification.py
```

## Model Information

- **Architecture**: VGG-style CNN with Batch Normalization
- **Training Dataset**: Food-101 (101,000 images)
- **Accuracy**: ~85% on test set
- **Input**: 128x128 RGB images
- **Output**: 101 food categories with softmax probabilities

## Troubleshooting

### Model Not Loading
- Ensure `food101_model.keras` exists in the project directory
- Check that TensorFlow and tf-keras are properly installed

### Out of Memory
- The Food-101 model is larger than the original
- Close other applications if memory issues occur

### Slow Prediction
- First prediction may be slower due to model loading
- Subsequent predictions will be faster

## Future Enhancements

Possible improvements:
1. Add Indian food categories (chapati, biryani, etc.)
2. Fine-tune on custom food datasets
3. Add nutritional information beyond calories
4. Implement object detection for multiple foods in one image
5. Add recipe suggestions based on classification

## Credits

- **Model**: Food-101 pre-trained model from Hugging Face
- **Architecture**: VGG-style CNN with Batch Normalization
- **Training**: 60,600 training images, 15,150 validation images
- **Source**: neck392/p4-food101-vggstyle-cnn-bn-labelsmoothing