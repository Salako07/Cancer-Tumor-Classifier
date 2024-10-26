# Cancer-Tumor-Classifier
### README for Tumor Classification Model

#### 1. **Project Overview**
   - **Objective:** This project aims to develop a deep learning image classification model that can accurately classify tumor types (Benign, Malignant, Normal) to support early diagnosis in healthcare.
   - **Significance:** Early and accurate tumor detection helps improve patient outcomes by aiding timely medical intervention.

#### 2. **Model Architecture**
   - **Model Type:** Convolutional Neural Network (CNN) with transfer learning.
   - **Base Architectures Used:**  
     - **DenseNet** (selected for final deployment due to higher accuracy)
     - **ResNet** (tested for comparison)
   - **Transfer Learning Rationale:** Using pre-trained models leverages existing knowledge from large image datasets, enabling higher accuracy even with limited medical images.

#### 3. **Dataset Information**
   - **Data Classes:** Benign (891), Malignant (420), Normal (134).
   - **Preprocessing Steps:** 
     - Resized and normalized images.
     - Class balancing through Image Augumentation.

#### 4. **Setup Instructions**
   - **Dependencies:** List key libraries (e.g., TensorFlow, Keras, OpenCV, Numpy, Pandas).
   - **Installation Steps:** :
     ```bash
     pip install -r requirements.txt
     ```
   - **Running the Model:** Steps to load data, train the model, and evaluate it.

#### 5. **Training and Evaluation**
   - **Training Process:** Explain your approach to model training:
   -  Transfer Learning: DenseNet and ResNet
       training epochs: DenseNet:15, ResNe:10,
       batch size: 32,
       weight: Imagenet.
   - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.
   - **Performance Summary:** DenseNet’s accuracy vs. ResNet for comparison.
   - DenseNet: ![DenseNet report](https://github.com/user-attachments/assets/9dfecd2a-d495-4c65-852b-38df6c8cc7e1)
   - ResNet:![ResNet](https://github.com/user-attachments/assets/a3988496-2632-4df8-b9ed-056fd3325983)



#### 6. **Results and Visualizations**
   - **Confusion Matrix:** ![image](https://github.com/user-attachments/assets/049312b3-2c3e-4b39-b783-322e62c96630) ![image](https://github.com/user-attachments/assets/215e73cc-0109-4ef4-8251-29cad2db9383)

.
   - **Sample Predictions:** Sample Input .![Benign](https://github.com/user-attachments/assets/67a4f4e2-7016-4bc3-b716-5e4ffb2b72e0)
   - Prediction:![Output_pred](https://github.com/user-attachments/assets/9a234cf2-63e8-4849-92be-244e2ab06bf8)



#### 7. **Usage**
   - **Inference:** Ongoing Deployment on Azure.

#### 8. **Future Work**
   - **Model Improvement:**  Additional Data willbe used to train the model, different acrchitectures wille considered.
   - **Extensions:** Possible applications or extensions, like multi-class tumor classification for additional tumor types.


