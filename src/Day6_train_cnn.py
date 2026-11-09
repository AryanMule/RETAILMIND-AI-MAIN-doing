
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.Day6_image_pipeline import (
    load_image_metadata,
    split_image_data,
    create_tf_dataset,
    get_augmentation_layer,
    IMAGE_SIZE
)


# --------------------------------------------------
# 1. Project paths and random seeds
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# --------------------------------------------------
# 2. Load image metadata
# --------------------------------------------------

df, class_names = load_image_metadata()

NUM_CLASSES = len(class_names)

print("\nNumber of classes:", NUM_CLASSES)
print("Classes:", class_names)


# --------------------------------------------------
# 3. Split training, validation and testing data
# --------------------------------------------------

train_df, test_df = split_image_data(df)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=train_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))


# --------------------------------------------------
# 4. Create TensorFlow datasets
# --------------------------------------------------

train_dataset = create_tf_dataset(
    train_df,
    training=True
)

val_dataset = create_tf_dataset(
    val_df,
    training=False
)

test_dataset = create_tf_dataset(
    test_df,
    training=False
)


# --------------------------------------------------
# 5. Calculate class weights
# --------------------------------------------------

classes = np.unique(train_df["label"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"]
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# --------------------------------------------------
# 6. Build CNN architecture
# --------------------------------------------------

augmentation = get_augmentation_layer()

model = keras.Sequential([

    layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    ),

    # Data augmentation
    augmentation,

    # Convolution block 1
    layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=2),

    # Convolution block 2
    layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=2),

    # Convolution block 3
    layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=2),

    # Classification layers
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# --------------------------------------------------
# 7. Compile CNN
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# --------------------------------------------------
# 8. Display model architecture
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 9. Configure callbacks
# --------------------------------------------------

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# --------------------------------------------------
# 10. Train CNN
# --------------------------------------------------

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=30,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# --------------------------------------------------
# 11. Save trained CNN
# --------------------------------------------------

model_path = MODELS_DIR / "fashion_cnn.keras"

model.save(model_path)


# --------------------------------------------------
# 12. Save class names
# --------------------------------------------------

with open(
    MODELS_DIR / "fashion_class_names.json",
    "w"
) as file:

    json.dump(class_names, file, indent=4)


# --------------------------------------------------
# 13. Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    MODELS_DIR / "day6_training_history.csv",
    index=False
)


# --------------------------------------------------
# 14. Print training summary
# --------------------------------------------------

print("\n========== CNN TRAINING COMPLETED ==========")

print("Best validation accuracy:",
      round(max(history.history["val_accuracy"]), 4))

print("Best validation loss:",
      round(min(history.history["val_loss"]), 4))

print("\nModel saved at:", model_path)

print("Class names saved successfully!")

print("Training history saved successfully!")
