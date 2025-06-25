


# Sickle Cell Detection Using Machine Learning

## Overview
This project implements a machine learning-based system for detecting sickle cell disease from blood cell images. The system uses multiple classification algorithms including Convolutional Neural Networks (CNN), Decision Trees, Random Forest, and Support Vector Machines (SVM) to classify blood cell images as either positive or negative for sickle cell disease.

## Dataset
- **Source**: [Google Drive Dataset](https://drive.google.com/file/d/1s3hEBguqyu4FqxWc-VDn_S8K4AJaLXiK/view?usp=drivesdk)
- **Classes**: 
  - `Negative`: Normal blood cells (originally labeled as "Clear")
  - `Positive`: Sickle cell affected blood cells (originally labeled as "Unlabeled")
- **Image Size**: 150x150 pixels
- **Total Samples**: 844 (after resampling)

## Architecture

### System Architecture
```
Input Images (150x150 RGB)
          ↓
   Data Preprocessing
          ↓
    Class Balancing (SMOTE)
          ↓
   Feature Extraction
          ↓
    Model Training
    ├── CNN Model
    ├── Decision Tree
    ├── Random Forest
    └── SVM
          ↓
   Model Evaluation
          ↓
    Prediction Output
```

### CNN Architecture
The Convolutional Neural Network consists of:
- **Input Layer**: 150x150x3 RGB images
- **Convolutional Layers**: 3 Conv2D layers with 32 filters each
- **Activation**: ReLU activation functions
- **Pooling**: MaxPooling2D layers (2x2)
- **Flattening**: Flatten layer to convert 2D to 1D
- **Dense Layers**: Configurable dense layers (0 in current setup)
- **Dropout**: 0.3 dropout rate for regularization
- **Output Layer**: Single neuron with sigmoid activation for binary classification

```
Conv2D(32, 3x3) → ReLU → MaxPooling2D(2x2)
       ↓
Conv2D(32, 3x3) → ReLU → MaxPooling2D(2x2)
       ↓
Conv2D(32, 3x3) → ReLU → MaxPooling2D(2x2)
       ↓
    Flatten
       ↓
   Dense(1) → Sigmoid
```

## Requirements

### Dependencies
```python
tensorflow
keras
opencv-python
numpy
matplotlib
scikit-learn
imbalanced-learn
seaborn
pickle
```

### Installation
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn imbalanced-learn seaborn
```

## Usage

### 1. Data Preprocessing
```python
# Set up directory and categories
Directory = "path/to/your/dataset"
Categories = ['Negative','Positive']
IMG_SIZE = 150

# Load and preprocess images
data = []
for category in Categories:
    folder = os.path.join(Directory, category)
    label = Categories.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img_arr)
        data.append([img_arr, label])
```

### 2. Class Balancing
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
x_res, y_res = smote.fit_resample(x.reshape(x.shape[0], -1), y)
```

### 3. Model Training

#### CNN Model
```python
# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# ... additional layers
model.add(Dense(1, activation='sigmoid'))

# Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', Precision(), Recall()])
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

#### Traditional ML Models
```python
# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train_flat, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_flat, y_train)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(x_train_flat, y_train)
```

### 4. Prediction
```python
def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Load model and predict
model = tf.keras.models.load_model("Our-work.keras")
prediction = model.predict([prepare('path/to/test/image.jpg')])
predicted_class = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
```

## Model Performance

### CNN Model
- **Training Accuracy**: ~83.5%
- **Validation Accuracy**: ~87.7%
- **Final Validation Loss**: 0.3248
- **Precision**: High precision (1.0 on validation)
- **Recall**: ~87.7%

### Traditional ML Models
- **Decision Tree Accuracy**: 62.3%
- **Random Forest Accuracy**: 69.3%
- **SVM Accuracy**: 70.2%


## Key Features

1. **Data Augmentation**: SMOTE oversampling to handle class imbalance
2. **Multiple Algorithms**: Comparison of CNN, Decision Tree, Random Forest, and SVM
3. **Visualization**: Training/validation curves and confusion matrices
4. **Model Persistence**: Saving trained models for future use
5. **Real-time Prediction**: Function to predict on new images

## Results Analysis

The CNN model significantly outperformed traditional machine learning algorithms:
- CNN achieved the highest accuracy (~87.7%)
- Perfect precision on validation set indicates low false positive rate
- SVM performed best among traditional ML methods (70.2%)
- Decision Tree had the lowest performance (62.3%)

## Future Improvements

1. **Data Augmentation**: Implement rotation, flipping, and scaling
2. **Advanced Architectures**: Try ResNet, DenseNet, or EfficientNet
3. **Transfer Learning**: Use pre-trained models for better feature extraction
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Cross-validation**: Implement k-fold cross-validation for robust evaluation
6. **Hyperparameter Tuning**: Optimize model parameters using grid search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers for the sickle cell image data
- TensorFlow and Keras communities for the deep learning framework
- Scikit-learn for traditional machine learning algorithms

## Contact

For questions or collaboration opportunities, please contact 

---

**Note**: This project is for educational and research purposes. For medical diagnosis, please consult healthcare professionals.
