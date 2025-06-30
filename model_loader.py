import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained emotion detection model
model = load_model("./Model/FER_DATA.keras")

# Emotion labels based on model's training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map emotions to music genres
map_emotion = {
    'angry': 'metal',
    'disgust': 'punk',
    'fear': 'electronic',
    'happy': 'pop',
    'sad': 'acoustic',
    'surprise': 'edm',
    'neutral': 'classical'
}

def predict_emotion(image_path: str) -> str:
    try:
        # Load and preprocess image (grayscale, resize, normalize)
        img = Image.open(image_path).convert('L').resize((48, 48))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)

        # Predict emotion
        predictions = model.predict(img_array, verbose=0)

        # Print raw predictions
        print("Raw prediction probabilities:", predictions.tolist()[0])

        emotion_index = int(np.argmax(predictions))
        emotion = emotion_labels[emotion_index].lower()
        print("Predicted emotion:", emotion)

        mapped_genre = map_emotion.get(emotion, "unknown")
        print("Mapped music genre:", mapped_genre)

        return mapped_genre

    except Exception as e:
        print("Error during prediction:", str(e))
        return "unknown"
