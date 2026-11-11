
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping


# Paths and settings
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fashion-product-images-small"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"

DAY6_MODEL = ROOT / "models" / "fashion_cnn.keras"
MODEL_PATH = ROOT / "models" / "fashion_vgg16.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5


# 1. Load dataset
df = pd.read_csv(CSV_PATH)

df = df[["id", "masterCategory"]].dropna()
df["id"] = pd.to_numeric(df["id"], errors="coerce")
df = df.dropna()

df["id"] = df["id"].astype(int)
df["image_path"] = df["id"].astype(str) + ".jpg"
df["image_path"] = df["image_path"].apply(
    lambda x: str(IMAGE_DIR / x)
)

df = df[df["image_path"].apply(os.path.exists)]

# Keep categories having at least 2 images
valid_categories = df["masterCategory"].value_counts()
valid_categories = valid_categories[valid_categories >= 2].index

df = df[df["masterCategory"].isin(valid_categories)]

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["masterCategory"])

X = df["image_path"].values
y = df["label"].values

num_classes = len(encoder.classes_)

print("Total images:", len(X))
print("Categories:", list(encoder.classes_))


# 2. Train-test-validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15,
    random_state=42, stratify=y_train
)


# 3. Image loading pipeline
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def make_dataset(paths, labels, preprocess=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if preprocess:
        dataset = dataset.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda x, y: (x / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(X_train, y_train, True)
val_ds = make_dataset(X_val, y_val, True)
test_ds = make_dataset(X_test, y_test, True)


# 4. Load pretrained VGG16
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 5. Train classification head
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

print("\nTraining classification head...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)


# 6. Fine-tune top VGG16 layers
print("\nFine-tuning top VGG16 layers...")

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    callbacks=[early_stop]
)


# 7. Evaluate VGG16 model
vgg_loss, vgg_accuracy = model.evaluate(test_ds)

print("\nVGG16 Test Accuracy:", round(vgg_accuracy * 100, 2), "%")


# 8. Compare with Day 6 CNN
if DAY6_MODEL.exists():
    print("\nEvaluating Day 6 CNN...")

    day6_model = load_model(DAY6_MODEL)

    # Day 6 CNN expects images normalized to 0-1
    day6_test_ds = make_dataset(X_test, y_test, False)

    _, day6_accuracy = day6_model.evaluate(day6_test_ds)

    print("Day 6 CNN Accuracy:", round(day6_accuracy * 100, 2), "%")
    print("Day 7 VGG16 Accuracy:", round(vgg_accuracy * 100, 2), "%")

    if vgg_accuracy > day6_accuracy:
        print("VGG16 performed better.")
    else:
        print("Day 6 CNN performed better.")

else:
    print("Day 6 model not found. Comparison skipped.")


# 9. Save upgraded model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
