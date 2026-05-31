# 🎵 Music Genre Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/librosa-Audio%20Processing-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-GTZAN-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>A deep learning pipeline that classifies music into 10 genres using Mel Spectrograms and a custom CNN — achieving 90.32% validation accuracy on the GTZAN dataset.</b>
</p>

---

## 📌 Overview

This project builds an end-to-end music genre classification system using Convolutional Neural Networks. Raw audio files are converted into **Mel Spectrograms** (2D visual representations of sound), which are then fed into a deep CNN trained to distinguish between 10 musical genres.

The model was trained on the widely-used **GTZAN Dataset** and achieves strong generalization, with a final validation accuracy of **90.32%** after 30 epochs.

---

## 🎯 Results

| Metric | Value |
|---|---|
| 🏋️ Training Accuracy | **98.09%** |
| ✅ Validation Accuracy | **90.32%** |
| 📉 Training Loss | 0.0650 |
| 📉 Validation Loss | 0.4193 |
| 🔁 Epochs | 30 |
| 📦 Batch Size | 32 |

---

## 🗂️ Project Structure

```
B241_B248_ML_MiniProject/
│
├── Final_Final_Train_Music_Genre_Classifier.ipynb   # Training pipeline
├── final_final_Test_Music_Genre.ipynb               # Inference on new audio
├── training_hist.json                               # Epoch-wise loss & accuracy logs
├── Trained_model.h5                                 # Saved model (Windows/Colab)
└── Trained_model.keras                              # Saved model (Mac/Linux)
```

---

## 🧠 Model Architecture

The CNN is built from scratch using **TensorFlow/Keras** with five convolutional blocks followed by dense layers:

```
Input (150×150×1 Mel Spectrogram)
│
├── Conv2D(32) → Conv2D(32) → MaxPool → 
├── Conv2D(64) → Conv2D(64) → MaxPool →
├── Conv2D(128) → Conv2D(128) → MaxPool → Dropout(0.3)
├── Conv2D(256) → Conv2D(256) → MaxPool →
├── Conv2D(512) → Conv2D(512) → MaxPool → Dropout(0.3)
│
├── Flatten
├── Dense(1200, relu) → Dropout(0.45)
└── Dense(10, softmax)   ← 10 genre classes
```

**Optimizer:** Adam (lr = 0.0001)  
**Loss:** Categorical Crossentropy

---

## 🎼 Genres Supported

| # | Genre | # | Genre |
|---|-------|---|-------|
| 0 | Blues | 5 | Jazz |
| 1 | Classical | 6 | Metal |
| 2 | Country | 7 | Pop |
| 3 | Disco | 8 | Reggae |
| 4 | Hip-Hop | 9 | Rock |

---

## 🔊 Audio Preprocessing Pipeline

Each audio file is processed as follows:

1. **Load** audio using `librosa` at original sample rate
2. **Chunk** the audio into **4-second segments** with **2-second overlap**
3. **Convert** each chunk to a **Mel Spectrogram**
4. **Resize** spectrogram to **150 × 150 pixels**
5. **Stack** all chunks → input tensor for the CNN
6. **Predict** genre per chunk → **majority vote** for final prediction

```python
# Prediction logic (majority vote over chunks)
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    return unique_elements[np.argmax(counts)]
```

---

## 📊 Training Curves

Training ran for **30 epochs**. The model converged smoothly with consistent validation improvement:

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 25.89% | 37.73% |
| 10 | 87.65% | 76.36% |
| 20 | 96.81% | 89.28% |
| 30 | **98.09%** | **90.32%** |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/music-genre-classification.git
cd music-genre-classification
```

### 2. Install dependencies

```bash
pip install tensorflow librosa numpy matplotlib seaborn scikit-learn kagglehub
```

### 3. Download the GTZAN Dataset

```python
import kagglehub
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
```

Or download directly from [Kaggle — GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

### 4. Train the model

Open and run `Final_Final_Train_Music_Genre_Classifier.ipynb` in **Google Colab** or Jupyter.

### 5. Test on your own audio

```python
# In final_final_Test_Music_Genre.ipynb
file_path = "your_song.mp3"
X_test = load_and_preprocess_data(file_path)
c_index = model_prediction(X_test)
print(f"Predicted Genre: {classes[c_index]}")
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `TensorFlow / Keras` | Model building & training |
| `librosa` | Audio loading & Mel spectrogram extraction |
| `NumPy` | Array & tensor operations |
| `scikit-learn` | Train/test split |
| `Matplotlib / Seaborn` | Visualization |
| `Google Colab` | Training environment (GPU) |
| `kagglehub` | Dataset download |

---

## 📁 Dataset

**GTZAN Music Genre Dataset**
- 1,000 audio tracks × 30 seconds each
- 10 genres × 100 tracks per genre
- Format: `.wav`, 22,050 Hz mono
- 📎 [Kaggle Link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## 📄 License

This project is for academic purposes under **NMIMS Mumbai**.  
Feel free to fork, modify, and build upon it with attribution.

---

<p align="center">Made with ❤️ and a lot of Mel Spectrograms</p>
