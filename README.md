# Face Recognition Eigenface
A complete implementation of face recognition using the Eigenfaces (PCA) method. This repository is structured to guide you through the transition from high-level code to low-level linear algebra.

## 📋 Core Components
We recommend following this order:

1.  **Tutorial :** Start with the interactive Jupyter Notebook to understand how 4,096 pixels are compressed into a tiny, searchable "Face Space" fingerprint.
2.  **Demo :** Run the local Python scripts to capture your own face and see the algorithm identify you in real-time alongside the Olivetti dataset.
3.  **Theory:** The "why" - explore the mathematical explanation of eigenvectors, and eigenvalues.

## 📂 Project Structure

| File | Category | Description |
| :--- | :--- | :--- |
| `eigenfaces_face_recognition_tutorial.ipynb` | **Tutorial** | **Start here.** Interactive walkthrough of PCA, data compression, and image reconstruction. |
| `data_capture.py` | **Demo** | Local utility to build your custom face dataset using your webcam. Can be skipped if you want to add faces into folder manually. |
| `face_recogniton_app.py` | **Demo** | The main real-time application that performs identity matching against the database. Requires faces in `./local_demo/data/` folder. |
| `eigenfaces_mathematical_explanation.mkd` | **Theory** | Technical deep-dive into mean centering, covariance matrices, and "Face Space" geometry. |

## 🚀 Getting Started
To run the real-time recognition demo, you must set up the project on your local machine. This allows the scripts to access your webcam and save your face data privately.

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone "https://github.com/henrysrtaylor/face-recognition-eigenface.git"
cd Face-Recognition-Eigenface
```

### 2. Set Up Virtual Environment
Windows:
```
python -m venv venv
.\venv\Scripts\activate
```

mac/linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Windows:
```
pip install -r requirements.txt
```

### Demo Execution Order
Once above steps have been followed and you wish to run the demo, follow these steps:

1. **Run the Capture:** `python data_capture.py`  
   * **What to expect:** A webcam window opens where you press **SPACE** to save cropped snapshots of your face into a local folder.
   * **The Point:** This provides the raw image data needed to "teach" the model your unique facial features.

2. **Run the Recognition:** `python face_recogniton_app.py`  
   * **What to expect:** The system processes all images into a mathematical "Face Space" and labels your live video feed as **"Authorized"** or **"Unknown"** based on visual similarity.
   * **The Point:** This demonstrates real-time identity verification by calculating the shortest distance between your live face and the stored training data.

> Note: If the system keeps labeling you as "Unknown," try to match the lighting and head angle of your original captures as closely as possible, as Eigenfaces are highly sensitive to these changes.

## 🧠 The "Face Space" Logic
The system works by projecting each face into a low-dimensional "fingerprint":
1.  **Mean Centering:** Subtract the "Mean Face" from the dataset to isolate unique individual features.
2.  **PCA Projection:** Images are projected onto **Eigenfaces** (ghostly face patterns) that represent primary modes of variation like lighting and structure.
3.  **Embedding:** Each face becomes a single coordinate in a subspace.
4.  **Matching:** Identification is achieved by calculating the **Euclidean Distance** between the live webcam or face query coordinate and stored database coordinates.

## ⚠️ Important: Demo Face Data
Real time face capture runs your default webcam and saves the images you capture to your local hardrive to allow the face recognition algorithm to run. There is also an option to skip this step by uploading faces to the `./local_demo/data/` folder, or skip the demo all together.

> Privacy Note: When you run the capture script, your images are saved only locally in the `./local_demo/data/` folder. No data is sent to any external server.