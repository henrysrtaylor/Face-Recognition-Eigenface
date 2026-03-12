"""
Simple Face Recognition System via Eigenfaces (PCA)
-------------------------------------------

Demo application performs real-time face recognition by comparing webcam feed against a combined dataset of the Olivetti faces and custom user data.

Note: Assumes that your local images (captured via data_capture.py or manually uploaded) are grayscale and sized (64, 64) to match the Olivetti dataset.

IMPORTANT: For best results, ensure your data capture was performed in the same  lighting conditions and environment as where you are running this application. Eigenfaces is sensitive to significant shifts in illumination and background.
"""

import os
import numpy as np
import cv2
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

your_data_path = "./local_demo/data"

# load other faces
olivetti_faces = fetch_olivetti_faces()
images = olivetti_faces.images
labels = olivetti_faces.target

# load your images from the collection 
full_paths = [f"{your_data_path}/{file_name}" for file_name in os.listdir(your_data_path) if file_name.endswith(('.png', '.jpg', '.jpeg'))]

error_string = f"Run data_capture.py or upload 10 greyscale 64x64 images of your face ({your_data_path}) first"
assert len(full_paths) > 0, FileExistsError(error_string)

your_images = []
for path in full_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32')
    if (img is not None) and (img.shape == (64, 64)):
        norm_img = cv2.normalize(img.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
        your_images.append(norm_img)
        del img; del norm_img # clean up

your_images = np.stack(your_images)
your_labels = np.array([-1 for face in your_images]) # create a temp label for you

images = np.concatenate([images, your_images])
labels = np.concatenate([labels, your_labels])

del your_images; del your_labels # clean up

images_flat = images.reshape(len(labels), 64*64) # flatten all images

# run Eigenfaces
mean_face = np.mean(images_flat, axis=0)
mean_centered_faces = images_flat - mean_face
pca = PCA(svd_solver='full').fit(mean_centered_faces)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

pca = PCA(n_components=0.95, whiten=True, svd_solver='full').fit(mean_centered_faces)
face_embeddings = pca.transform(mean_centered_faces)


# Run active app which will identify when specifically you is in frame, based on the data you uploaded
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
window_name = "Face Capture - Press SPACE to Save, ESC to Exit"
img_counter = 1

while True:
    ret, frame = cam.read()
    if not ret: break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(110, 110)
    )

    # Identify face on webcam
    display_frame = frame.copy()
    for (x, y, w, h) in faces:
        face_crop = gray_image[y:y+h, x:x+w] # Crop to face only
        face_resized = cv2.resize(face_crop, (64, 64)) # Resize to match Olivetti (64x64)
        face_flat = face_resized.reshape(64*64).reshape(1, -1) / 255.0 # flatten all images

        # Check if identified you by looking at distances for all faces
        new_face_centered = face_flat - mean_face 
        new_face_embedding = pca.transform(new_face_centered.reshape(1, -1))
        distances = euclidean_distances(new_face_embedding, face_embeddings).flatten()
        database_index = int(np.argmin(distances))
        database_label = int(labels[database_index])
#        
        # Set colour based on whether Eigenfaces identified you. Green for you, Red for not.
        colours = (0, 255, 0) if database_label == -1 else (0, 0, 255)
        name_label = "Authorized" if database_label == -1 else "Unknown"

        # Add face outline on webcam
        cv2.putText(
            display_frame, 
            name_label, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8,         # Font scale
            colours,     # Matches your rectangle color
            2,           # Thickness
            cv2.LINE_AA  # Anti-aliasing for smoother text
        )
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), colours, 2)

        # Display nearest image from database bottom left
        h_f, w_f, _ = display_frame.shape
        y1, y2, x1, x2 = h_f - 74, h_f - 10, 10, 74 # Added 10px padding from corner
        cv2.putText(display_frame, "DB Match", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        try:
            raw_match = (images[database_index] * 255).astype(np.uint8)
            nearest_image_match = cv2.cvtColor(raw_match, cv2.COLOR_GRAY2BGR)  # get nearest database match between your data and olivetti dataset
            display_frame[y1:y2, x1:x2] = nearest_image_match
        except Exception as e:
            display_frame[y1:y2, x1:x2] = np.zeros((64, 64, 3))

    cv2.imshow(window_name, display_frame)

    k = cv2.waitKey(1)
    if k%256 == 27: # ESC
        break

cam.release()
cv2.destroyAllWindows()