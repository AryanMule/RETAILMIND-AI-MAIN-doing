
from pathlib import Path

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


# 1. Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "fashion-product-images-small"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "styles.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
RANDOM_STATE = 42


# 2. Load metadata and prepare image paths
def load_image_metadata():

    print("Loading product metadata...")

    df = pd.read_csv(
        CSV_PATH,
        on_bad_lines="skip"
    )

    # Keep only required columns
    df = df[["id", "masterCategory"]].copy()

    # Remove missing values
    df = df.dropna(subset=["id", "masterCategory"])

    # Convert IDs to integers
    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    )

    df = df.dropna(subset=["id"])

    df["id"] = df["id"].astype(int)

    # Remove duplicate product IDs
    df = df.drop_duplicates(subset=["id"])

    # Convert category into string
    df["masterCategory"] = df["masterCategory"].astype(str)

    # Build image paths
    df["image_path"] = df["id"].apply(
        lambda product_id: str(
            IMAGE_DIR / f"{product_id}.jpg"
        )
    )

    # Keep only records whose images exist
    df = df[
        df["image_path"].apply(
            lambda path: Path(path).is_file()
        )
    ].copy()

    # Encode category labels
    class_names = sorted(
        df["masterCategory"].unique().tolist()
    )

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    df["label"] = df["masterCategory"].map(class_to_index)

    df = df.reset_index(drop=True)

    print("\nDataset loaded successfully!")
    print("Total usable images:", len(df))
    print("Number of categories:", len(class_names))
    print("\nCategory distribution:")
    print(df["masterCategory"].value_counts())

    return df, class_names


# 3. Split into training and testing data
def split_image_data(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nTraining images:", len(train_df))
    print("Testing images:", len(test_df))

    return train_df, test_df


# 4. Load and preprocess a single image
def load_and_preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# 5. Create TensorFlow dataset
def create_tf_dataset(df, training=False):

    image_paths = df["image_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 10000),
            seed=RANDOM_STATE
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# 6. Image augmentation layer
def get_augmentation_layer():

    augmentation = tf.keras.Sequential([

        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.1),

        tf.keras.layers.RandomZoom(0.1)

    ], name="image_augmentation")

    return augmentation


# 7. Main pipeline function
def prepare_image_pipeline():

    df, class_names = load_image_metadata()

    train_df, test_df = split_image_data(df)

    train_dataset = create_tf_dataset(
        train_df,
        training=True
    )

    test_dataset = create_tf_dataset(
        test_df,
        training=False
    )

    augmentation = get_augmentation_layer()

    print("\nImage pipeline prepared successfully!")

    return (
        train_dataset,
        test_dataset,
        class_names,
        augmentation
    )


# 8. Run directly for pipeline verification
if __name__ == "__main__":

    train_ds, test_ds, class_names, augmentation = (
        prepare_image_pipeline()
    )

    print("\nClass names:", class_names)

    for images, labels in train_ds.take(1):

        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)

        # Test augmentation on a batch
        augmented_images = augmentation(
            images,
            training=True
        )

        print(
            "Augmented batch shape:",
            augmented_images.shape
        )
