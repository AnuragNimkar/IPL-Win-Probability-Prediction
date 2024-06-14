import streamlit as st
import pickle
from PIL import Image
import numpy as np

# Function to load the model


@st.cache
def load_model(model_path):
    with open(model_path, 'rb') as f:
        # model = pickle.load(f)
        model = pickle.load(
            open("C:/Users/Lenovo/Desktop/DSBDA MINI PROJECT/pipe.pkl", 'rb'))
    return model

# Function to make predictions


def predict_disease(model, image):
    # Preprocess the image (resize, normalize, etc.)
    # Replace this with your preprocessing code
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)

    return prediction

# Function to preprocess the image


def preprocess_image(image):
    # Replace this with your preprocessing code
    # Example: resizing the image to match the model's input size
    processed_image = image.resize((224, 224))
    processed_image = np.array(processed_image) / 255.0  # Normalize

    return processed_image


def main():
    st.title("Disease Prediction App")

    # Sidebar - File Upload
    st.sidebar.title("Upload Model")
    model_file = st.sidebar.file_uploader("Upload Pickle File", type=['pkl'])
    if model_file is not None:
        model = load_model(model_file)
        st.sidebar.success("Model successfully loaded.")

    # Main content
    st.write("### Drag and Drop Image Here")
    image_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Display the uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model_file is not None and st.sidebar.button("Predict"):
            # Make prediction
            prediction = predict_disease(model, image)
            st.write("### Prediction:", prediction)
        elif model_file is None:
            st.sidebar.warning("Please upload a model file first.")


if __name__ == "__main__":
    main()
