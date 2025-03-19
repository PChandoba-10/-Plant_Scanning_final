import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("plant_model1.h5")

# Ensure this variable exists for import
plant_classification_model = model

# Define soil classes
PLANT_CLASSES = [
   'aloevera', 
   'arjun', 
   'ashwagandha', 
   'babool', 
   'bael', 
   'bakuchi',
   'barberry', 
   'bhilawa', 
   'bhringraj', 
   'chilly', 
   'coffee', 
   'coriander',
    'curry', 
    'giloy', 
    'ginger',
    'glochidion', 
    'gotu kola', 
    'hibiscus', 
    'jasmine', 
    'lemon', 
    'madar', 
    'mango', 
    'marigold', 
    'mint', 
    'moringa', 
    'naruneendi', 
    'neem', 
    'onion', 
    'papaya', 
    'ricinus', 
    'rose', 
    'sarpagandha', 
    'shatavari', 
    'stereoserpum', 
    'tomato', 
    'tulsi', 
    'turmeric', 
    'wedelia'
    ]

def predict_plant(image_array):
    """Predicts plant type from an image array"""
    img = tf.image.resize(image_array, [256, 256])  # Normalize
    img = img / 255.0  # Resize for model
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return PLANT_CLASSES[prediction.argmax()]
