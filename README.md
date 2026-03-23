# 🧠 AI-Based Brain Tumor Detection and Analysis System

An intelligent medical imaging system that detects and analyzes brain tumors from MRI scans using deep learning. The system uses a **ResNet-50 Convolutional Neural Network (CNN)** for tumor classification and **Grad-CAM visualization** to highlight tumor regions, providing explainable AI insights for medical analysis.

---

# 🚀 Features

- 🧠 **Brain Tumor Classification**
  - Detects tumor types from MRI images
  - Classes: Glioma, Meningioma, Pituitary, No Tumor

- 🔍 **Explainable AI Visualization**
  - Grad-CAM heatmaps highlight tumor regions

- 📊 **Tumor Analysis**
  - Tumor segmentation
  - Tumor size estimation
  - Tumor depth analysis

- 📈 **Growth Prediction**
  - Predicts tumor progression over time

- 🩺 **Prognosis Insights**
  - Survival trend visualization
  - Diagnosis confidence metrics

- 🌐 **Web Application Interface**
  - Upload MRI scans
  - Interactive results dashboard

---

# 🏗 System Architecture

The system follows a pipeline architecture:

MRI Image Upload  
↓  
Image Preprocessing  
↓  
ResNet-50 CNN Model  
↓  
Tumor Classification  
↓  
Grad-CAM Visualization  
↓  
Tumor Segmentation & Analysis  
↓  
Results Dashboard  

---

# 🧠 Algorithms Used

### 1️⃣ ResNet-50 CNN with Transfer Learning
- Deep convolutional neural network for image classification
- Extracts complex features from MRI scans
- Classifies tumors into four categories

### 2️⃣ Grad-CAM (Explainable AI)
- Generates heatmaps showing tumor regions
- Helps interpret the model's predictions

---

# 🛠 Technologies Used

| Category | Technology |
|--------|-------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Model Architecture | ResNet-50 CNN |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Frontend | HTML, JavaScript, TailwindCSS |
| Image Processing | PIL, NumPy, OpenCV |
| Visualization | Matplotlib |

---

# 📂 Project Structure

brain-tumor-detection-ai

│

├── models/

│      trained_model.pth

│

├── templates/

│      index.html

│      results.html

│

├── static/

│      css/

│      js/

│

├── app.py

├── requirements.txt

└── README.md

---

# ⚙️ Installation

### 1️⃣ Clone the repository

git clone https://github.com/YOUR_USERNAME/brain-tumor-detection-ai.git

cd brain-tumor-detection-ai

---

### 2️⃣ Create virtual environment

python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

---

### 3️⃣ Install dependencies

pip install -r requirements.txt

---

# ▶️ Running the Application

Start the web server:

python app.py

Open in browser:

http://localhost:8000

Upload an MRI image to analyze tumor detection results.

---

# 📊 Example Output

The system provides:

- Tumor type prediction
- Confidence score
- Grad-CAM heatmap visualization
- Tumor segmentation results
- Tumor growth prediction graphs
- Prognosis insights

---

# 📌 Use Cases

- AI-assisted medical image analysis
- Research in medical imaging
- Educational deep learning projects
- AI healthcare applications

---

# ⚠️ Disclaimer

This project is intended for **research and educational purposes only** and should **not be used for real clinical diagnosis** without professional medical validation.

---

# 👨‍💻 Author

Developed as part of a **Mini Project in Artificial Intelligence and Data Science**.

---

# ⭐ Future Improvements

- Integration with larger MRI datasets
- Real-time hospital system integration
- Advanced segmentation using U-Net
- Deployment as a cloud AI service

---

# 📜 License

This project is open-source and available under the MIT License.
