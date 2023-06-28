import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
import torch
import os
import glob
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


# Function to display similar images with scores
def display_similar_images(image_list, score_list):
    for i in range(len(image_list)):
        image = Image.open(image_list[i])
        cols = st.columns(2)
        cols[0].image(image, caption=image_list[i], use_column_width=True)
        cols[1].write(f"Similarity score : {score_list[i]}")

def imageInput(src):
    # Load pretrained model
    load_pretrained_model_weight = ResNet50(weights='imagenet')

    # Use this to extract features before the final layer
    feature_extractor = Model(inputs=load_pretrained_model_weight.input,
                    outputs=load_pretrained_model_weight.get_layer("avg_pool").output)

    # Load similarity data
    cos_similarities = pd.read_csv('simi.csv', index_col=0)

    if src == 'Upload your own Hairstyle Image':
        image_file = st.file_uploader("Upload Your desired hairstyle image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Hairstyle Image', use_column_width=True)
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='hairstyle_1st/weights/best.pt', force_reload=True)  
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)
            #--Display predicton
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='AI Hairstyle Recommender', use_column_width=True)

            # Button to trigger the similarity prediction
            if st.button('Predict Similar Hairstyles'):
                if 'uploaded_image' in cos_similarities.columns:
                    cos_similarities = cos_similarities.drop(columns=['uploaded_image'])
                    cos_similarities = cos_similarities.drop(index=['uploaded_image'])
                
                # Preprocess uploaded image
                load_resize_orginal_image = img.resize((224, 224))
                numpy_image_array = img_to_array(load_resize_orginal_image)
                image_batch = np.expand_dims(numpy_image_array, axis=0)
                processed_image = preprocess_input(image_batch.copy())

                # Extract features from processed_image
                new_image_features = feature_extractor.predict(processed_image)

                # Append new image features to the existing ones
                existing_image_features = np.load('images_feature.npy')
                updated_image_features = np.vstack((existing_image_features, new_image_features))

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


    elif src == 'From sample Hairstyle Images': 
        # Image selector slider
        imgpath = glob.glob('C:/Users/46058007/OneDrive - MMU/Attachments/Stylebook/Hair_recommendation_app/dataset/images/train/*') #should be the same as what the similarity algorithm was trained on
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        # image_select = cos_similarities.columns[imgsel]
        submit = st.button("Predict Hairstyle type")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                model = torch.hub.load('ultralytics/yolov5','custom', path= 'hairstyle_1st/weights/best.pt', force_reload=True) 
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='AI Hairstyle Recommendation')


        # Button to trigger the similarity prediction
        if st.button('Suggest Similar Hairstyles'):
            
            image_select = cos_similarities.columns[imgsel]
            
            
            
            # Fetch and sort similar images
            similar_images = cos_similarities[image_select].sort_values(ascending=False)[1:11].index
            similar_images_scores = cos_similarities[image_select].sort_values(ascending=False)[1:11].values

            # # Display original image
            # original = load_img(image_select, target_size=(500, 500))
            # st.image(original, caption="Selected Image", use_column_width=True)

            st.write("-"*100)
            st.header("Images with similar features or colours:")

            # Display similar images
            display_similar_images(similar_images, similar_images_scores)


def main():
    
    st.image("logo.JPG", width = 500)
    st.title("Stylebook Directory Limited")
    st.header("AI Tool for Hairstyle Recommendation")
    st.header("👈🏽 Select the Image Source options")
    st.sidebar.title('⚙️Options')
    src = st.sidebar.radio("Select input source.", ['From sample Hairstyle Images', 'Upload your own Hairstyle Image'])
    imageInput(src)
   
if __name__ == '__main__':
    
    main()
