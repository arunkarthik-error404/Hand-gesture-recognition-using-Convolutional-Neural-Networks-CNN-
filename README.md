## Hand gesture recognition using Convolutional Neural Networks (CNN)

## 📌 Objective
Develop a Convolutional Neural Network (CNN) model to classify hand gestures representing American Sign Language (ASL) letters (A–Z, excluding J and Z) using grayscale images.

## 🧠 Background
American Sign Language (ASL) is widely used by the deaf and hard-of-hearing community. This project aims to automate the recognition of hand gestures to aid communication using deep learning techniques, specifically CNNs, which are known for their effectiveness in image classification tasks.

## 📁 Dataset
- **Format**: Grayscale images, 28x28 pixels.
- **Labels**: 0-25, corresponding to A–Z (excluding 9=J and 25=Z).
- **Source**: Provided ZIP file (hand_sign.zip).

## 🧱 CNN Architecture
- **Input Layer**: 28x28 grayscale image
- **Conv Layer 1**: Custom kernel/filter, ReLU activation
- **Conv Layer 2**: Custom kernel/filter, ReLU activation
- **Max Pooling Layer**: 2x2 or 3x3 pooling window
- **Flatten Layer**
- **Fully Connected (Dense) Layer**
- **Dropout Layer**: Rate = 0.2
- **Output Layer**: 26 nodes with Softmax activation

## ⚙️ Model Training
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10
- **Evaluation**: Track loss and accuracy over epochs

## 📉 Results & Observations
- Plotted training loss and testing accuracy per epoch.
- Analysis on overfitting and regularization using dropout layers.
- Discussed how CNN handles:
  - **Weight sharing**
  - **Translation invariance**

## 📊 Success Metrics
- **Primary**: Accuracy
- **Secondary**: Loss minimization

## 🖼️ Model Visualization
Diagrammatic representation of the CNN architecture is provided in the project files.

## 📦 Requirements
- Python
- PyTorch
- NumPy
- Matplotlib (for plotting)
- Jupyter Notebook (optional)

## 🧪 Run Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gesture-recognition-cnn.git
   cd gesture-recognition-cnn
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:

   ```bash
   python train.py
   ```
4. Evaluate/test:

   ```bash
   python evaluate.py
   ```

## 📚 References

* [PyTorch Documentation](https://pytorch.org/docs/stable/nn.html)
* [Saving and Loading Models in PyTorch](https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE)
* [Handwritten Character Recognition Research Paper](https://www.researchgate.net/publication/379382573_HANDWRITTEN_CHARACTER_RECOGNITION_IN_ASSYRIAN_LANGUAGE_USING_CONVOLUTIONAL_NEURAL_NETWORK)

---

> This project demonstrates the power of deep learning in bridging communication gaps and improving accessibility through gesture recognition.

