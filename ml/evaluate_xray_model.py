import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATHS
# =========================
MODEL_PATH = "ml/models/xray_model.hdf5"
TEST_DIR = "D:/my_projects/dataset_xray/chest_xray/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD TEST DATA
# =========================
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # IMPORTANT
)

# =========================
# PREDICTIONS
# =========================
preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes

# =========================
# METRICS
# =========================
print("\n📊 CLASSIFICATION REPORT:\n")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

print("\n📉 CONFUSION MATRIX:\n")
print(confusion_matrix(y_true, y_pred))

# =========================
# ACCURACY
# =========================
accuracy = np.mean(y_pred == y_true)
print(f"\n✅ FINAL TEST ACCURACY: {accuracy * 100:.2f}%")