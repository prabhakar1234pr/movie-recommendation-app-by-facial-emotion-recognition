import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import random

# Load the trained model
model = load_model(r'venv\emotion_detection_model.keras')

# Emotion categories
emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Movie recommendations
movie_recommendations = {
    'anger': [
        'Joker (2019)', '12 Angry Men (1957)', 'Falling Down (1993)', 'American History X (1998)',
        'The Revenant (2015)', 'There Will Be Blood (2007)', 'Gladiator (2000)',
        'Kill Bill: Volume 1 (2003)', 'Mad Max: Fury Road (2015)', 'The Godfather Part II (1974)'
    ],
    'disgust': [
        'Requiem for a Dream (2000)', 'Trainspotting (1996)', 'The Fly (1986)', 'Se7en (1995)',
        'A Clockwork Orange (1971)', 'Oldboy (2003)', 'The Girl with the Dragon Tattoo (2011)',
        'Mother! (2017)', 'Nightcrawler (2014)', 'Silence of the Lambs (1991)'
    ],
    'fear': [
        'The Exorcist (1973)', 'Hereditary (2018)', 'Get Out (2017)', 'The Shining (1980)',
        'It Follows (2014)', 'A Quiet Place (2018)', 'Psycho (1960)',
        'Alien (1979)', 'The Witch (2015)', 'Midsommar (2019)'
    ],
    'happy': [
        'Forrest Gump (1994)', 'The Pursuit of Happyness (2006)', 'La La Land (2016)', 'Up (2009)',
        'Amélie (2001)', 'Little Miss Sunshine (2006)', 'The Grand Budapest Hotel (2014)',
        "Singin' in the Rain (1952)", 'Paddington 2 (2017)', 'Zootopia (2016)'
    ],
    'sad': [
        "Schindler's List (1993)", 'Grave of the Fireflies (1988)', 'Manchester by the Sea (2016)',
        'The Green Mile (1999)', 'Blue Valentine (2010)', 'Atonement (2007)',
        'Brokeback Mountain (2005)', 'Marley & Me (2008)', 'Coco (2017)',
        'Requiem for a Dream (2000)'
    ],
    'surprise': [
        'Forrest Gump (1994)', 'The Pursuit of Happyness (2006)', 'La La Land (2016)', 'Up (2009)',
        'Amélie (2001)', 'Little Miss Sunshine (2006)', 'The Grand Budapest Hotel (2014)',
        "Singin' in the Rain (1952)", 'Paddington 2 (2017)', 'Zootopia (2016)'
    ],
    'neutral': [
        'Forrest Gump (1994)', 'The Pursuit of Happyness (2006)', 'La La Land (2016)', 'Up (2009)',
        'Amélie (2001)', 'Little Miss Sunshine (2006)', 'The Grand Budapest Hotel (2014)',
        "Singin' in the Rain (1952)", 'Paddington 2 (2017)', 'Zootopia (2016)'
    ]
}

# Function to predict emotion from an image
def predict_emotion(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))
    predictions = model.predict(img)
    max_index = np.argmax(predictions[0])
    return emotions[max_index]

# Streamlit app
st.title("Emotion-Based Movie Recommendation App")

# Camera input
img_file = st.camera_input("Take a picture")

if img_file:
    # Convert file to an OpenCV image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Predict emotion
    emotion = predict_emotion(img)
    
    # Display the image and prediction
    st.image(img, channels="BGR")
    st.write(f"The predicted emotion is: {emotion}")
    
    # Recommend a random movie based on the predicted emotion
    st.write("Movie Recommendation:")
    recommended_movie = random.choice(movie_recommendations[emotion])
    st.write(recommended_movie)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Take a picture using the camera input above to detect the emotion and get a movie recommendation.")

