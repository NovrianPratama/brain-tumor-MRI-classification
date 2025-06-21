# 🧠 Brain Tumor Classification using MRI Images

This project focuses on classifying brain tumors using MRI images into four categories using deep learning. The aim is to support early detection of brain tumors, which plays a vital role in determining effective treatment and improving patient outcomes.

---

## 📚 About the Dataset

The dataset used is from Kaggle: [**Brain Tumor MRI Dataset**](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

This dataset is a combination of:

- **Figshare**
- **SARTAJ Dataset** *(some glioma images were removed due to mislabeling)*
- **Br35H**

### 📊 Dataset Details:

- 📷 **Total Images**: 7023 brain MRI scans
- 🧠 **Classes**:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor (from Br35H)

> ⚠️ *Note: The image sizes vary. During preprocessing, all images are resized and unnecessary margins are removed to enhance model accuracy and generalization.*

---

## 🧪 Methods

This project uses a **Convolutional Neural Network (CNN)** architecture to:

- Detect and classify brain tumors into four distinct categories
- Train a single multi-task model rather than multiple individual models
- Export the model in multiple formats for deployment

The full training and evaluation process is documented in the notebook: `brain_tumor_classification.ipynb`.

---

## 🏗️ Model Deployment

After training, the model was exported in the following formats:

- ✅ **SavedModel** (`/models/saved_model`)
- ✅ **TensorFlow Lite** (`/models/tflite`)
- ✅ **TensorFlow.js** (`/models/tfjs`)

---

## 📁 Project Structure
```
├───tfjs_model             # TensorFlow.js model files
|   ├───group1-shard1of1.bin
|   └───model.json
├───tflite                 # TensorFlow Lite model
|   ├───model.tflite
|   └───label.txt
├───saved_model            # Saved model format
|   ├───variables
|   |   ├───variables.data-00000-of-00001
|   |   └───variables.index
|   ├───saved_model.pb
|   └───variables
├── Submission_Akhir_Novrian_Pratama.ipynb  # Complete workflow
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation
```

---

## 🚀 How to Run

1. Clone the repository and open the notebook.
2. Upload your `kaggle.json` API token to access the dataset.
3. Run the notebook to:
   - Download and preprocess the dataset
   - Train and evaluate the CNN model
   - Export the model in multiple formats

---

## 📎 Additional Notes

- The model achieved over **85% accuracy** on both training and testing datasets.
- Visualization of accuracy and loss across epochs is included in the notebook.
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks were used to optimize performance and prevent overfitting.

---

## 🔗 Dataset Link

> [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

---

## ✉️ Contact

If you have questions or suggestions, feel free to contact:

**Novrian Pratama**  
📧 novrianprtama2@gmail.com

---
