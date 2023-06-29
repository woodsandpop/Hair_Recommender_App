import streamlit as st
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# Function to display similar images with scores
def display_similar_images(image_list, score_list):
    for i in range(len(image_list)):
        image = Image.open(image_list[i])
        cols = st.columns(2)
        cols[0].image(image, caption=image_list[i], use_column_width=True)
        cols[1].write(f"Similarity score : {score_list[i]}")

# Load similarity data
cos_similarities = pd.read_csv('simi.csv', index_col=0)

# Load pretrained model
load_pretrained_model_weight = ResNet50(weights='imagenet')

# Use this to extract features before the final layer
feature_extractor = Model(inputs=load_pretrained_model_weight.input,
                   outputs=load_pretrained_model_weight.get_layer("avg_pool").output)

# Streamlit code
st.title('Image Similarity')
st.write('##')

# Choosing between existing and new image
option = st.radio('Choose an option', ('Use existing image', 'Upload new image'))

if option == 'Use existing image':
    st.subheader('Option 1: Select an existing image')
    st.write('Use the slider to select an image')
    image_index = st.slider('Image index', 0, len(cos_similarities.columns)-1)
    image_select = cos_similarities.columns[image_index]
    
elif option == 'Upload new image':
    st.subheader('Option 2: Upload a new image')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if 'uploaded_image' in cos_similarities.columns:
            cos_similarities = cos_similarities.drop(columns=['uploaded_image'])
            cos_similarities = cos_similarities.drop(index=['uploaded_image'])

        # Preprocess uploaded image
        load_resize_orginal_image = image.resize((224, 224))
        numpy_image_array = img_to_array(load_resize_orginal_image)
        image_batch = np.expand_dims(numpy_image_array, axis=0)
        processed_image = preprocess_input(image_batch.copy())

        # Extract features from processed_image
        new_image_features = feature_extractor.predict(processed_image)

        # Append new image features to the existing ones
        existing_image_features = np.load('images_feature.npy')
        updated_image_features = np.vstack((existing_image_features, new_image_features))
        # np.save('images_feature.npy', updated_image_features)

        # Update similarity dataframe and csv
        new_similarities = cosine_similarity(updated_image_features)
        updated_cos_similarities = pd.DataFrame(new_similarities, columns=cos_similarities.columns.append(pd.Index(['uploaded_image'])), index=cos_similarities.columns.append(pd.Index(['uploaded_image'])))
        updated_cos_similarities.to_csv('simi.csv')

        st.write("-"*100)
        st.header("Similar images to the uploaded image:")

        # Fetch and sort similar images to the uploaded one
        similar_images = updated_cos_similarities.loc['uploaded_image'].sort_values(ascending=False)[1:11].index
        similar_images_scores = updated_cos_similarities.loc['uploaded_image'].sort_values(ascending=False)[1:11].values



        # Display similar images
        display_similar_images(similar_images, similar_images_scores)

if st.button('Find Similar Images'):
    # Fetch and sort similar images
    similar_images = cos_similarities[image_select].sort_values(ascending=False)[1:11].index
    similar_images_scores = cos_similarities[image_select].sort_values(ascending=False)[1:11].values

    # Display original image
    original = load_img(image_select, target_size=(500, 500))
    st.image(original, caption="Selected Image", use_column_width=True)

    st.write("-"*100)
    st.header("Images with similar features or colours:")

    # Display similar images
    display_similar_images(similar_images, similar_images_scores)
