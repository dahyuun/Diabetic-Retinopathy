#copyright@dahyun mok

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf  # Import TensorFlow
import random
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the ensemble model
ensemble_model = tf.keras.models.load_model('0821_1_E.h5')

# Define the test data directories
original_test_dir = './data/original/EyePACS/test'
clahe_test_dir = './data/clahe/EyePACS/test'

# Load test labels
test_labels = pd.read_csv('./data/original/EyePACS/testlabels.csv')

# 테스트 이미지를 로드하고 전처리하는 함수
def load_and_preprocess_images(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# 테스트 이미지를 로드하고 전처리
test_images_224 = []
test_images_128 = []
true_labels = []
not_found_images = []

for index, row in test_labels.iterrows():
    image_name = row['image']
    level = row['level']
    true_labels.append(level)

    # 원본 테스트 이미지를 로드하고 전처리
    found = False
    for ext in ['.jpg', '.jpeg', '.tif']:
        original_image_path = os.path.join(original_test_dir, image_name + ext)
        if os.path.exists(original_image_path):
            img = load_and_preprocess_images(original_image_path, (224, 224))
            test_images_224.append(img)
            found = True
            break
    
    if not found:
        not_found_images.append(image_name)

    # CLAHE 테스트 이미지를 로드하고 전처리
    found = False
    for ext in ['.jpg', '.jpeg', '.tif']:
        clahe_image_path = os.path.join(clahe_test_dir, image_name + ext)
        if os.path.exists(clahe_image_path):
            img = load_and_preprocess_images(clahe_image_path, (128, 128))
            test_images_128.append(img)
            found = True
            break

    if not found:
        not_found_images.append(image_name)

# 찾을 수 없는 이미지 이름 출력
if not_found_images:
    print("이미지를 찾을 수 없습니다:")
    print(not_found_images)

# # 넘파이 배열로 변환
# test_images_224 = np.array(test_images_224)
# test_images_128 = np.array(test_images_128)

# # 앙상블 모델로 예측
# predictions_224, predictions_128 = ensemble_model.predict([test_images_224, test_images_128])

# # Take the average of predictions as the final prediction
# final_predictions = (predictions_224 + predictions_128) / 2

# 이미지 배치로 나누어 예측
batch_size = 32
num_images = len(test_images_224)
final_predictions = []

for i in range(0, num_images, batch_size):
    batch_images_224 = test_images_224[i:i+batch_size]
    batch_images_128 = test_images_128[i:i+batch_size]
    batch_predictions_224, batch_predictions_128 = ensemble_model.predict([batch_images_224, batch_images_128])
    batch_final_predictions = (batch_predictions_224 + batch_predictions_128) / 2
    final_predictions.extend(batch_final_predictions)

final_predictions = np.array(final_predictions)

# Convert probabilities to binary predictions (0 or 1)
binary_predictions = (final_predictions > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(true_labels, binary_predictions)
precision, recall, fscore, _ = score(true_labels, binary_predictions)
sensitivity = recall[1]
specificity = recall[0]
fpr, tpr, _ = roc_curve(true_labels, final_predictions)
roc_auc = auc(fpr, tpr)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Sensitivity (True Positive Rate):", sensitivity)
print("Specificity (True Negative Rate):", specificity)
print("AUC:", roc_auc)

# Randomly select and display example images
random_indices = random.sample(range(len(test_images_224)), 4)

plt.figure(figsize=(15, 8))
for i, index in enumerate(random_indices, 1):
    image_224 = test_images_224[index]
    image_128 = test_images_128[index]
    true_label = true_labels[index]
    predicted_label = binary_predictions[index][0]

    # Display original image
    plt.subplot(4, 4, i*4-3)
    plt.imshow(image_224)
    plt.title('Original Image (224x224)')
    plt.axis('off')

    # Display resized image
    plt.subplot(4, 4, i*4-2)
    plt.imshow(image_128)
    plt.title('Resized Image (128x128)')
    plt.axis('off')

    # Display true label
    plt.subplot(4, 4, i*4-1)
    plt.text(0.5, 0.5, f'True Label: {true_label}', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')

    # Display predicted label
    plt.subplot(4, 4, i*4)
    plt.text(0.5, 0.5, f'Predicted Label: {predicted_label}', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()