# Research Paper Implementation using Pre-trained ResNet50

## 📚 Course Information
- **Course Name:** Deep Learning / Machine Learning
- **Lab Title:** Research Paper Implementation with Pre-trained Model
- **Model Used:** ResNet50 (Pre-trained on ImageNet)
- **Dataset Used:** CIFAR-10

---

## 📄 Research Paper Details

**Paper Title:** Deep Residual Learning for Image Recognition  
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Conference:** CVPR 2016  
**Paper Link:**  
https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html  

---

## 📊 Dataset Information

**Dataset:** CIFAR-10  
**Description:**
- 60,000 RGB images
- 10 classes
- 32x32 resolution
- 50,000 training images
- 10,000 testing images

**Dataset Link:**  
https://www.cs.toronto.edu/~kriz/cifar.html  

---

## 🎯 Project Objective

- Study the ResNet architecture.
- Implement transfer learning using a pre-trained ResNet50 model.
- Fine-tune top layers for CIFAR-10 classification.
- Evaluate model performance.
- Compare results with original research paper.

---

## ⚙️ Methodology

### 1️⃣ Preprocessing
- Images normalized to range [0,1]
- Resized to 224x224
- Labels converted to categorical format
- Dataset loaded using TensorFlow

### 2️⃣ Transfer Learning
- Loaded ResNet50 with ImageNet weights
- Removed top classification layer
- Added custom Dense layers
- Froze base layers
- Fine-tuned classification head

### 3️⃣ Hyperparameters

| Parameter | Value |
|------------|--------|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 5 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

---

## 📈 Performance Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 📊 Results

- Training Accuracy: ~80–85%
- Validation Accuracy: ~75–82%
- Test Accuracy: ~78–85%

---

## 📉 Comparison with Research Paper

| Metric | Research Paper | Our Implementation |
|---------|----------------|--------------------|
| Accuracy | ~90%+ (ImageNet) | ~80% (CIFAR-10) |
| Model Depth | 50+ Layers | ResNet50 |
| Dataset Size | Large-scale ImageNet | CIFAR-10 |

---

## 🚀 Improvements Suggested

- Increase number of training epochs
- Unfreeze deeper layers for fine-tuning
- Apply data augmentation
- Use learning rate scheduling
- Train on larger dataset

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## 📂 Project Structure

