#copyright@dahyun mok
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adamax

import os
import cv2
import shutil

# 데이터 경로 설정
train_data_dir1 = './data/original/train/0'
gray_data_dir1 = './data/clahe/train/0'
train_data_dir2 = './data/original/train/1'
gray_data_dir2 = './data/clahe/train/1'
train_data_dir3 = './data/original/test'
gray_data_dir3 = './data/clahe/test'

# CLAHE 적용 함수 정의
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    return clahe_image

# 이미지 파일 목록 얻기
image_files1 = [f for f in os.listdir(train_data_dir1) if os.path.isfile(os.path.join(train_data_dir1, f))]
image_files2 = [f for f in os.listdir(train_data_dir2) if os.path.isfile(os.path.join(train_data_dir2, f))]
image_files3 = [f for f in os.listdir(train_data_dir3) if os.path.isfile(os.path.join(train_data_dir3, f))]

# 이미지 처리 및 복사
for image_file in image_files1:
    image_path = os.path.join(train_data_dir1, image_file)
    image = cv2.imread(image_path)
    
    # CLAHE 적용
    clahe_image = apply_clahe(image)
    
    # 저장할 경로 생성
    gray_image_path = os.path.join(gray_data_dir1, image_file)
    
    # 그레이스케일 이미지 저장
    cv2.imwrite(gray_image_path, clahe_image)
for image_file in image_files2:
    image_path = os.path.join(train_data_dir2, image_file)
    image = cv2.imread(image_path)
    
    # CLAHE 적용
    clahe_image = apply_clahe(image)
    
    # 저장할 경로 생성
    gray_image_path = os.path.join(gray_data_dir2, image_file)
    
    # 그레이스케일 이미지 저장
    cv2.imwrite(gray_image_path, clahe_image)
for image_file in image_files3:
    image_path = os.path.join(train_data_dir3, image_file)
    image = cv2.imread(image_path)
    
    # CLAHE 적용
    clahe_image = apply_clahe(image)
    
    # 저장할 경로 생성
    gray_image_path = os.path.join(gray_data_dir3, image_file)
    
    # 그레이스케일 이미지 저장
    cv2.imwrite(gray_image_path, clahe_image)

print("Image processing and copying complete.")

# 데이터 경로 설정
original_train_dir = './data/original/train'
clahe_train_dir = './data/clahe/train'
img_height, img_width = 224, 224
batch_size = 32
validation_split = 0.125  # Validation 데이터의 비율 설정

# 데이터 로드 및 전처리
datagen = ImageDataGenerator(
    validation_split=validation_split,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

train_generator = datagen.flow_from_directory(
    original_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'  # 학습 데이터로 설정
)

validation_generator = datagen.flow_from_directory(
    original_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    subset='validation'  # 검증 데이터로 설정
)

# CLAHE 이미지 데이터 로드 및 전처리
clahe_datagen = ImageDataGenerator(
    validation_split=validation_split,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

clahe_train_generator = clahe_datagen.flow_from_directory(
    clahe_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'  # 학습 데이터로 설정
)

clahe_validation_generator = clahe_datagen.flow_from_directory(
    clahe_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    subset='validation'  # 검증 데이터로 설정
)

# ResNet50 모델 로드
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# 이미지 입력을 위한 Input 레이어 생성
image_input = Input(shape=(img_height, img_width, 3))

# Original 이미지와 CLAHE 이미지를 각각 처리하는 브랜치 생성
original_processed = base_model(image_input)
clahe_processed = base_model(image_input)

# Global Average Pooling 레이어 추가
original_processed = GlobalAveragePooling2D()(original_processed)
clahe_processed = GlobalAveragePooling2D()(clahe_processed)

# 두 브랜치의 출력을 결합하여 Concatenate 레이어 생성
concatenated = Concatenate()([original_processed, clahe_processed])

# Binary Classification을 위한 Fully Connected 레이어 추가
dropout_rate = 0.4  # Dropout 비율 설정
x = Dropout(dropout_rate)(concatenated)  # Dropout 레이어 추가
predictions = Dense(1, activation='sigmoid')(x)

# 새로운 모델 생성
model = Model(inputs=image_input, outputs=predictions)

# 모델 컴파일
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 학습 수행
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]
)

# 학습 과정에서의 Loss와 Accuracy를 출력
print("Train Loss:", history.history['loss'])
print("Train Accuracy:", history.history['accuracy'])
print("Validation Loss:", history.history['val_loss'])
print("Validation Accuracy:", history.history['val_accuracy'])

# Loss와 Accuracy 그래프 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training and Validation Metrics')
plt.show()

# 학습된 모델 저장
model.save('resnet_0821_1.h5')
