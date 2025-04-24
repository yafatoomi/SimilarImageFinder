import os
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from PIL import Image
import streamlit as st
import random

# Sample categories and products (For generating CSV)
categories = ['Stationery', 'Remote', 'Clothes', 'Cars']
stationery_items = ['Pen', 'Pencil', 'Notebook', 'Eraser', 'Marker', 'Highlighter', 'Ruler', 'Stapler', 'Paper', 'Clipboard']
remote_items = ['TV Remote', 'AC Remote', 'Fan Remote', 'DVD Remote', 'Projector Remote']
clothes_items = ['Shirt', 'T-shirt', 'Jeans', 'Jacket', 'Sweater', 'Skirt', 'Trousers', 'Shorts', 'Hat', 'Scarf']
car_items = ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe', 'Truck', 'Minivan']

# Generate random data for the CSV
def generate_sample_data():
    product_ids = [f'PID_{i+1}' for i in range(1000)]
    product_names = []
    category_list = []
    product_links = []

    for i in range(1000):
        category = random.choice(categories)
        if category == 'Stationery':
            product_name = random.choice(stationery_items)
        elif category == 'Remote':
            product_name = random.choice(remote_items)
        elif category == 'Clothes':
            product_name = random.choice(clothes_items)
        elif category == 'Cars':
            product_name = random.choice(car_items)

        product_names.append(f'{product_name} {i+1}')
        category_list.append(category)
        product_links.append(f'http://example.com/{category}/{product_name.replace(" ", "-").lower()}')

    data = {
        'Product ID': product_ids,
        'Product Name': product_names,
        'Category': category_list,
        'Image Link': product_links
    }

    df = pd.DataFrame(data)
    df.to_csv('product_data.csv', index=False)
    print("CSV file with 1000 products generated successfully!")
    return df

# Load VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

# Preprocessing image
def prepare_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Extract features using VGG16
def extract_features(image_path):
    preprocessed_image = prepare_image(image_path)
    features = model.predict(preprocessed_image)
    return features.flatten()

# Extract features for all images in folder
def extract_features_from_folder(folder_path):
    feature_list = []
    image_names = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            features = extract_features(image_path)
            feature_list.append(features)
            image_names.append(image_name)
    return feature_list, image_names

# Find top visually similar images
def find_top_similar_images(query_image, feature_list, image_names, folder_path, top_n=5):
    query_features = extract_features(query_image)
    feature_shape = feature_list[0].shape[0]
    if query_features.shape[0] != feature_shape:
        print(f"Shape mismatch: query_features.shape = {query_features.shape[0]}, feature_list[0].shape = {feature_shape}")
        return []
    similarities = []
    for i, features in enumerate(feature_list):
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities.append((image_names[i], similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [os.path.join(folder_path, name) for name, _ in similarities[:top_n]]

# Chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "what" in user_input and ("this app" in user_input or "app do" in user_input):
        return "Elevated Vision helps you find visually similar images using deep learning and a camera or image upload."
    elif "how" in user_input and "work" in user_input:
        return "This app uses a pre-trained VGG16 model to extract visual features from images. Then it compares them using similarity scores."
    elif "step" in user_input or "how to use" in user_input:
        return (
            "Here are the steps:\n"
            "1. Upload a CSV with image URLs.\n"
            "2. The app downloads and processes the images.\n"
            "3. Go to 'Find Similar Images' page and use your camera to take a picture.\n"
            "4. The app finds and shows the top 5 most similar images."
        )
    elif "vgg16" in user_input:
        return "VGG16 is a deep learning model trained to recognize image features. It's used here to compare image similarity."
    elif "who" in user_input and ("created" in user_input or "made" in user_input):
        return "This app was created by Muhammed Hasan Mohiuddin Ghori & Amaara Fatima."
    elif "help" in user_input:
        return "Try asking me things like 'what does this app do', 'what is VGG16', or 'how do I use the app'."
    else:
        return "I'm not sure about that—try asking how the app works, what it's for, or how to use it."

# Page 1: Introduction + Chatbot
def introduction_page():
    st.markdown(""" 
    <h1 style='text-align:center; color:#2c3e50;'>🤖 Elevated Vision Assistant</h1>
    <h4 style='text-align:center; color:#2980b9;'>By Muhammed Hasan Mohiuddin Ghori & Amaara Fatima</h4>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💬 Ask Me Anything About Elevated Vision")
    st.markdown("I'm your friendly assistant! Ask me questions like:")
    st.markdown("- What does this app do?")
    st.markdown("- How does it work?")
    st.markdown("- What are the steps?")
    st.markdown("- What is VGG16?")
    st.markdown("- Who made this app?")
    user_input = st.text_input("Ask a question:")
    if user_input:
        response = chatbot_response(user_input)
        st.markdown(f"**Bot:** {response}")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Next"):
            st.session_state.page = "Upload & Process Data"

# Page 2: Upload + Process CSV
def upload_and_process_page():
    st.markdown("""
    <h1 style='text-align:center; color:#3498db;'>ELEVATED VISION</h1>
    <h4 style='text-align:center; color:#3498db;'>By Muhammed Hasan Mohiuddin Ghori & Amaara Fatima</h4>
    """, unsafe_allow_html=True)

    st.subheader("IDEAL STAGE: Upload CSV File")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.success("CSV loaded successfully!")
    else:
        st.info("No file uploaded. Generating sample CSV...")
        df = generate_sample_data()
        st.success("Sample CSV generated successfully!")

    st.write("CSV Columns:", df.columns.tolist())

    # Auto-detect image URL column
    url_column = None
    for col in df.columns:
        if "url" in col.lower() or "link" in col.lower():
            url_column = col
            break
    if not url_column:
        st.error("No column with image URLs found.")
        return

    st.subheader("CREATION STAGE: Downloading Images")
    output_dir = "downloaded_images"
    os.makedirs(output_dir, exist_ok=True)

    image_url_map = {}

    if len(os.listdir(output_dir)) == 0:
        st.write("Downloading images...")
        for _, row in df.iterrows():
            image_url = row[url_column]
            image_name = f"{row['Product ID']}.jpg"
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    output_path = os.path.join(output_dir, image_name)
                    with open(output_path, "wb") as file:
                        file.write(response.content)
                    image_url_map[image_name] = image_url
            except Exception:
                st.error(f"Failed to download image {image_url}")
        st.success("All images downloaded.")
    else:
        st.info("Images already downloaded. Skipping download.")
        for _, row in df.iterrows():
            image_name = f"{row['Product ID']}.jpg"
            image_url_map[image_name] = row[url_column]

    st.session_state.image_url_map = image_url_map

    st.subheader("REFINEMENT STAGE: Extracting Features")
    if 'feature_list' not in st.session_state:
        st.write("Extracting features, please wait...")
        feature_list, image_names = extract_features_from_folder(output_dir)
        st.session_state.feature_list = feature_list
        st.session_state.image_names = image_names
        st.success("Feature extraction completed.")
    else:
        st.info("Features already extracted.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.page = "Introduction"
    with col2:
        if st.button("Next"):
            st.session_state.page = "Find Similar Images"

# Page 3: Camera input + results
def camera_query_page():
    st.title("📸 Find Visually Similar Images")

    if 'feature_list' not in st.session_state or 'image_names' not in st.session_state:
        st.warning("Please complete the previous steps first on 'Upload & Process Data' page.")
        return
    if 'image_url_map' not in st.session_state:
        st.warning("Image URL map not found. Please re-run the image upload process.")
        return

    captured_image = st.camera_input("Take a picture to find similar images")

    if captured_image is not None:
        query_image_path = "temp_query_image.jpg"
        with open(query_image_path, "wb") as f:
            f.write(captured_image.getbuffer())

        st.image(Image.open(query_image_path), caption="Query Image")

        st.write("Finding similar images...")
        top_similar_images = find_top_similar_images(
            query_image_path,
            st.session_state.feature_list,
            st.session_state.image_names,
            "downloaded_images"
        )

        st.success("Top 5 similar images found:")
        columns = st.columns(5)
        for i, img_path in enumerate(top_similar_images):
            image_name = os.path.basename(img_path)
            image_url = st.session_state.image_url_map.get(image_name, "URL not available")
            with columns[i]:
                st.image(img_path, width=150)
                st.markdown(f"[🔗 View Image]({image_url})", unsafe_allow_html=True)

    if st.button("Back"):
        st.session_state.page = "Upload & Process Data"

# Main
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Introduction"

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", [
        "Introduction",
        "Upload & Process Data",
        "Find Similar Images"
    ])

    if st.session_state.page == "Introduction":
        introduction_page()
    elif st.session_state.page == "Upload & Process Data":
        upload_and_process_page()
    elif st.session_state.page == "Find Similar Images":
        camera_query_page()

if __name__ == "__main__":
    main()
