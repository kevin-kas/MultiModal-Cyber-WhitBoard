import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
# Set random seed for reproducibility
np.random.seed(42)

def load_images_and_labels(folder_path):
    """Load image data without data augmentation"""
    images = []
    labels = []
    class_names = [d for d in os.listdir(folder_path)
                   if os.path.isdir(os.path.join(folder_path, d))]

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        img_names = [f for f in os.listdir(class_folder)
                     if not f.startswith('.') and '.ipynb' not in f]

        for img_name in img_names:
            img_path = os.path.join(class_folder, img_name)
            try:
                # Basic image processing
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((45, 45))

                images.append(np.array(img).flatten())
                labels.append(class_name)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    return np.array(images), np.array(labels)


# Load data
print("Loading training data...")
train_images, train_labels = load_images_and_labels('./train_data')
print(f"Training data loaded, {len(train_images)} samples in total")

print("Loading test data...")
test_images, test_labels = load_images_and_labels('./test_data')
print(f"Test data loaded, {len(test_images)} samples in total")

# Process labels
if train_labels.ndim > 1:
    train_labels = train_labels.ravel()
if test_labels.ndim > 1:
    test_labels = test_labels.ravel()

le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# Feature standardization
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

# Feature dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
train_images_pca = pca.fit_transform(train_images_scaled)
test_images_pca = pca.transform(test_images_scaled)

print(f"Feature dimensionality reduced from {train_images.shape[1]} to {train_images_pca.shape[1]}")

# Build more complex model selection and hyperparameter optimization
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
    'learning_rate_init': [0.001, 0.0001]
}

base_mlp = MLPClassifier(random_state=42, max_iter=100, early_stopping=True, validation_fraction=0.1)

# Use stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
print("Starting hyperparameter optimization...")
grid_search = GridSearchCV(
    base_mlp,
    param_grid,
    scoring='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=2
)

grid_search.fit(train_images_pca, train_labels_encoded)
best_mlp = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the model
train_accuracy = best_mlp.score(train_images_pca, train_labels_encoded)
test_accuracy = best_mlp.score(test_images_pca, test_labels_encoded)

print(f"Training set accuracy: {train_accuracy:.4f}")
print(f"Test set accuracy: {test_accuracy:.4f}")

# Detailed classification report
test_predictions = best_mlp.predict(test_images_pca)
print("\nClassification Report:")
print(classification_report(
    test_labels_encoded,
    test_predictions,
    target_names=le.classes_
))

# Confusion matrix visualization
cm = confusion_matrix(test_labels_encoded, test_predictions)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Save the model and related components
joblib.dump(best_mlp, 'mlp_model.pkl')
joblib.dump(scaler, 'mlp_scaler.pkl')
joblib.dump(pca, 'mlp_pca.pkl')
joblib.dump(le, 'mlp_label_encoder.pkl')

print("Model and preprocessing components saved")

def predict_single_image(image_path):
    """Load the saved model and predict a single image"""
    # Load saved components
    loaded_model = joblib.load('mlp_model.pkl')
    loaded_scaler = joblib.load('mlp_scaler.pkl')
    loaded_pca = joblib.load('mlp_pca.pkl')
    loaded_le = joblib.load('mlp_label_encoder.pkl')

    # Load and preprocess the image
    img = Image.open(image_path).convert('L')
    img = img.resize((45, 45))
    img_array = np.array(img).flatten().reshape(1, -1)

    # Feature standardization and dimensionality reduction
    img_scaled = loaded_scaler.transform(img_array)
    img_pca = loaded_pca.transform(img_scaled)

    # Prediction
    prediction_encoded = loaded_model.predict(img_pca)
    prediction_class = loaded_le.inverse_transform(prediction_encoded)

    return prediction_class[0]

sample_prediction = predict_single_image('./test_data/1/exp79.jpg')
print(f"Sample prediction result: {sample_prediction}")