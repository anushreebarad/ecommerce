

import certifi
from dotenv import load_dotenv
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from io import BytesIO
# MongoDB imports
from pymongo import MongoClient
import hashlib
from urllib.parse import quote_plus




# Load environment variables from .env file
load_dotenv()

# Constants
TRAINED_DB_PATH = "db"

# MongoDB Atlas connection using safe encoding
MONGO_USER = os.environ.get("MONGO_USER", "anushree")
MONGO_PASS = os.environ.get("MONGO_PASS", "Anu@123")
MONGO_CLUSTER = os.environ.get("MONGO_CLUSTER", "ecommerce.msffomx.mongodb.net")
DB_NAME = "ecommerce"
USERS_COLLECTION = "users"

# Safely encode username and password
MONGO_USER_ENC = quote_plus(MONGO_USER)
MONGO_PASS_ENC = quote_plus(MONGO_PASS)
MONGO_URI = f"mongodb+srv://{MONGO_USER_ENC}:{MONGO_PASS_ENC}@{MONGO_CLUSTER}/?retryWrites=true&w=majority&appName=ecommerce"

# SSL Certificate setup
os.environ['SSL_CERT_FILE'] = certifi.where()


# MongoDB Atlas connection
client = MongoClient(MONGO_URI, tls=True)
db = client[DB_NAME]
users_col = db[USERS_COLLECTION]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# --- Helper Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    if users_col.find_one({"username": username}):
        return False, "Username already exists."
    users_col.insert_one({"username": username, "password": hash_password(password)})
    return True, "Signup successful!"

def login(username, password):
    user = users_col.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return True, "Login successful!"
    return False, "Invalid username or password."


def show_login_signup():
    st.markdown("""
        <h2 style='text-align:center; color:#884EA0;'>Login / Signup</h2>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,3,2])
    with col2:
        choice = st.radio("Select action", ["Login", "Signup"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if choice == "Signup":
            if st.button("Signup"):
                ok, msg = signup(username, password)
                st.success(msg) if ok else st.error(msg)
        else:
            if st.button("Login"):
                ok, msg = login(username, password)
                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success(msg)
                else:
                    st.error(msg)

def show_products():
    st.header("Products")
    img_files = [f for f in os.listdir(TRAINED_DB_PATH) if f.lower().endswith((".jpg", ".jpeg"))]
    cols = st.columns(4)
    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(TRAINED_DB_PATH, img_file)
        with cols[idx % 4]:
            st.image(img_path, caption=img_file, use_column_width=True)

def init_session_state():
    if "feature_vectors" not in st.session_state:
        st.session_state.feature_vectors = None
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = None


@st.cache_resource
def load_model() -> tf.keras.Model:
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path: Union[str, BytesIO], model: tf.keras.Model) -> Union[np.ndarray, None]:
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img).flatten()
        tf.keras.backend.clear_session()
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        tf.keras.backend.clear_session()
        return None

def predict_class_label(image_path: Union[str, BytesIO], model: tf.keras.Model) -> str:
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        # Use the full ResNet50 model for classification
        from tensorflow.keras.applications.resnet50 import ResNet50 as ResNet50Full
        classifier = ResNet50Full(weights='imagenet')
        class_preds = classifier.predict(preprocessed_img)
        decoded = decode_predictions(class_preds, top=1)[0][0]
        label = f"{decoded[1]} ({decoded[2]*100:.2f}%)"
        tf.keras.backend.clear_session()
        return label
    except Exception as e:
        return f"Prediction failed: {str(e)}"

@st.cache_data(show_spinner=False)
def get_feature_vectors_from_db(db_path: str, model: tf.keras.Model) -> tuple[np.ndarray, list[str]]:
    feature_list = []
    image_paths = []
    try:
        for img_path in os.listdir(db_path):
            if img_path.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(db_path, img_path)
                features = extract_features(path, model)
                if features is not None:
                    feature_list.append(features)
                    image_paths.append(path)
        feature_vectors = np.vstack(feature_list)
        return feature_vectors, image_paths
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return np.array([]), []

def find_similar_images(image_path: Union[str, BytesIO], feature_vectors: np.ndarray, image_paths: list[str],
                        model: tf.keras.Model, threshold: float = 0.5, top_n: int = 5) -> list[str]:
    query_features = extract_features(image_path, model)
    if query_features is None:
        return []
    similarities = cosine_similarity([query_features], feature_vectors)
    similarities_indices = [i for i in range(len(similarities[0])) if similarities[0][i] > threshold]
    similarities_indices = sorted(similarities_indices, key=lambda i: similarities[0][i], reverse=True)
    similar_images = [image_paths[i] for i in similarities_indices[:top_n]]
    tf.keras.backend.clear_session()
    return similar_images

# --- Main App ---

# --- Multipage E-commerce App ---
def page_home():
    st.markdown("""
        <h1 style='text-align: center; color: #2E86C1;'>E-commerce Visual Search</h1>
        <p style='text-align: center;'>Welcome to our modern e-commerce platform. Browse products, search visually, and enjoy a seamless experience!</p>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1515168833906-d2a3b82b1a48?auto=format&fit=crop&w=800&q=80", use_column_width=True)

def page_products():
    st.markdown("<h2 style='color:#117A65;'>Product Gallery</h2>", unsafe_allow_html=True)
    show_products()

def page_visual_search():
    st.markdown("<h2 style='color:#B9770E;'>Visual Search</h2>", unsafe_allow_html=True)
    st.write("Upload an image and find similar products from our database. Powered by ResNet50 and AI.")
    init_session_state()
    model = load_model()
    if st.session_state.feature_vectors is None:
        with st.spinner("Loading product database..."):
            st.session_state.feature_vectors, st.session_state.image_paths = get_feature_vectors_from_db(
                TRAINED_DB_PATH, model)
            st.success("Database loaded successfully!")
    uploaded_img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
    if uploaded_img_file is not None:
        uploaded_img = Image.open(uploaded_img_file)
        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Predicting class label..."):
            label = predict_class_label(uploaded_img_file, model)
            st.info(f"Predicted class: {label}")
        with st.spinner("Extracting features..."):
            query_features = extract_features(uploaded_img_file, model)
            if query_features is not None:
                st.success("Features extracted successfully!")
            else:
                st.error("Failed to extract features from the uploaded image.")
                return
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
        top_n = st.slider("Number of Similar Products", 1, 10, 5)
        if st.button("Find Similar Products"):
            with st.spinner("Searching for similar products..."):
                similar_images = find_similar_images(
                    uploaded_img_file,
                    st.session_state.feature_vectors,
                    st.session_state.image_paths,
                    model,
                    threshold,
                    top_n
                )
                if similar_images:
                    st.success("Similar products found!")
                    for i, similar_image in enumerate(similar_images):
                        image = Image.open(similar_image)
                        st.image(image, caption=f"Similar Product {i + 1}", use_column_width=True)
                        st.write("")
                else:
                    st.write("No similar products found!")
                tf.keras.backend.clear_session()

def page_login():
    st.markdown("<h2 style='color:#884EA0;'>Login / Signup</h2>", unsafe_allow_html=True)
    show_login_signup()

def page_profile():
    st.markdown(f"<h2 style='color:#2874A6;'>Profile</h2>", unsafe_allow_html=True)
    st.write(f"Welcome, **{st.session_state.get('username', 'User')}**!")
    st.write("(Profile features coming soon.)")


def main():
    st.set_page_config(page_title="E-commerce Visual Search", layout="wide")
    # Sidebar branding
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1170/1170678.png", width=80)
    st.sidebar.title("E-commerce Visual Search")
    st.sidebar.markdown("---")

    # Authentication flow: Only show login/signup until logged in
    if not st.session_state.get("logged_in"):
        st.sidebar.markdown("#### Please log in to continue")
        page_login()
        st.sidebar.caption("© 2025 E-commerce Visual Search")
        return

    # After login, show full navigation
    pages = {
        "Home": page_home,
        "Products": page_products,
        "Visual Search": page_visual_search,
        "Profile": page_profile
    }
    page = st.sidebar.radio("Go to", list(pages.keys()))
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Logged in as: {st.session_state.get('username','User')}")
    st.sidebar.caption("© 2025 E-commerce Visual Search")
    # Render selected page
    pages[page]()

if __name__ == "__main__":
    main()