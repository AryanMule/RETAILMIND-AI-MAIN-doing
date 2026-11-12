"""
Day 8 - Segmentation, Background Removal and Grad-CAM
RetailMind AI

This script reuses the VGG16 classifier trained on Day 7 and adds:

1. Product classification
2. Grad-CAM explainability
3. Product segmentation
4. Background removal
5. Uniform white catalog background
6. Bounding-box localization
7. End-to-end CV pipeline

Run from the project root:

    python src/Day8_segmentation_gradcam.py

Optional custom image:

    python src/Day8_segmentation_gradcam.py data/test_product.jpg
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

# __file__ points to:
# RetailMind_AI/src/Day8_segmentation_gradcam.py
#
# parent gives:
# RetailMind_AI/src
#
# parent.parent gives:
# RetailMind_AI
ROOT = Path(__file__).resolve().parent.parent

# Fashion dataset folder used in Day 7
DATA_DIR = ROOT / "data" / "fashion-product-images-small"

# Product images folder
IMAGE_DIR = DATA_DIR / "images"

# Dataset metadata file
CSV_PATH = DATA_DIR / "styles.csv"

# Day 7 trained VGG16 model
MODEL_PATH = ROOT / "models" / "fashion_vgg16.keras"

# All Day 8 outputs will be stored here
OUTPUT_DIR = ROOT / "outputs" / "day8"

# VGG16 input size used in your Day 7 file
IMG_SIZE = (128, 128)

# Last convolution layer of VGG16
# Grad-CAM uses this layer to understand important image regions.
LAST_CONV_LAYER_NAME = "block5_conv3"


# =========================================================
# 2. VALIDATE REQUIRED FILES
# =========================================================

def validate_project_files():
    """
    Check whether the dataset, images and Day 7 model exist.

    The Day 8 script cannot work without:
    - styles.csv
    - images folder
    - trained Day 7 VGG16 model
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"styles.csv was not found at:\n{CSV_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory was not found at:\n{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nDay 7 model was not found.\n"
            f"Expected model path:\n{MODEL_PATH}\n\n"
            "Run Day 7 first to generate fashion_vgg16.keras."
        )

    # Create the output folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. REBUILD DAY 7 CLASS NAMES
# =========================================================

def load_class_names():
    """
    Recreate the class names used by LabelEncoder on Day 7.

    Day 7 used:

        encoder = LabelEncoder()
        encoder.fit_transform(df["masterCategory"])

    LabelEncoder stores category names in sorted order.

    Because the Day 7 script did not save LabelEncoder separately,
    this function reconstructs the same category order from styles.csv.
    """

    category_counts = {}

    with open(CSV_PATH, mode="r", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            product_id = row.get("id")
            category = row.get("masterCategory")

            # Skip rows with missing values
            if not product_id or not category:
                continue

            try:
                # Day 7 converted IDs to numbers and then integers
                numeric_id = int(float(product_id))
            except ValueError:
                continue

            image_path = IMAGE_DIR / f"{numeric_id}.jpg"

            # Day 7 retained only rows whose image existed
            if not image_path.exists():
                continue

            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

    # Day 7 kept categories containing at least two valid images
    valid_categories = [
        category
        for category, count in category_counts.items()
        if count >= 2
    ]

    # LabelEncoder sorts string classes
    valid_categories.sort()

    if not valid_categories:
        raise ValueError(
            "No valid masterCategory values were found."
        )

    return valid_categories


# =========================================================
# 4. FIND A DEFAULT TEST IMAGE
# =========================================================

def get_default_image_path():
    """
    Return the first valid JPG image from the dataset.

    This allows the script to run even when the user does not
    provide an image path on the command line.
    """

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No JPG images were found inside:\n{IMAGE_DIR}"
        )

    return image_paths[0]


# =========================================================
# 5. LOAD AN IMAGE WITH OPENCV
# =========================================================

def load_image(image_path):
    """
    Read an image using OpenCV.

    OpenCV loads color images in BGR format.

    Returns:
        image: NumPy array with shape:
               height x width x 3
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image was not found:\n{image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"OpenCV could not read the image:\n{image_path}"
        )

    return image


# =========================================================
# 6. PREPARE IMAGE FOR THE DAY 7 VGG16 MODEL
# =========================================================

def prepare_image_for_vgg(image):
    """
    Prepare an OpenCV image for the VGG16 model.

    Your Day 7 model expects:
        128 x 128 x 3

    Your Day 7 training pipeline used:
        preprocess_input(image)

    The same preprocessing must be used during prediction.
    """

    # OpenCV reads BGR, but the Day 7 TensorFlow image loader
    # produced RGB images.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the same dimensions used during Day 7 training
    resized_image = cv2.resize(
        rgb_image,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert to float32 before VGG16 preprocessing
    image_array = resized_image.astype(np.float32)

    # Apply the same VGG16 preprocessing used on Day 7
    image_array = preprocess_input(image_array)

    # Add batch dimension:
    # 128 x 128 x 3 becomes 1 x 128 x 128 x 3
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# =========================================================
# 7. CLASSIFY PRODUCT
# =========================================================

def classify_product(model, model_input, class_names):
    """
    Predict the product's masterCategory.

    Returns:
        class_index
        class_name
        confidence
        all_probabilities
    """

    predictions = model.predict(
        model_input,
        verbose=0
    )[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    if class_index < len(class_names):
        class_name = class_names[class_index]
    else:
        class_name = f"Class {class_index}"

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": confidence,
        "all_probabilities": predictions
    }


# =========================================================
# 8. CREATE GRAD-CAM HEATMAP
# =========================================================

def make_gradcam_heatmap(
    model_input,
    model,
    last_conv_layer_name=LAST_CONV_LAYER_NAME,
    class_index=None
):
    """
    Generate a Grad-CAM heatmap.

    Grad-CAM identifies the image regions that influenced the
    selected classification result.

    Grad-CAM uses:
        - Output of the last convolution layer
        - Gradient of the predicted class score
        - Importance weight of each feature map
    """

    try:
        last_conv_layer = model.get_layer(
            last_conv_layer_name
        )
    except ValueError as error:
        available_conv_layers = [
            layer.name
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]

        raise ValueError(
            f"\nLayer '{last_conv_layer_name}' was not found.\n"
            f"Available Conv2D layers: {available_conv_layers}"
        ) from error

    # Create a model that returns:
    # 1. Last convolution layer's output
    # 2. Final classification prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # GradientTape records operations so TensorFlow can
    # calculate the gradients afterward.
    with tf.GradientTape() as tape:

        convolution_outputs, predictions = grad_model(
            model_input,
            training=False
        )

        # If a class was not supplied, explain the class
        # having the highest predicted probability.
        if class_index is None:
            class_index = tf.argmax(predictions[0])

        # Score of the class being explained
        selected_class_score = predictions[:, class_index]

    # Find how the class score changes with respect to the
    # last convolution layer's feature maps.
    gradients = tape.gradient(
        selected_class_score,
        convolution_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    # Average each feature map's gradient across its
    # height and width.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove the batch dimension
    convolution_outputs = convolution_outputs[0]

    # Give more importance to feature maps that had
    # stronger gradients for the selected class.
    heatmap = tf.reduce_sum(
        convolution_outputs * pooled_gradients,
        axis=-1
    )

    # Negative values do not positively support the class
    heatmap = tf.maximum(heatmap, 0)

    maximum_value = tf.reduce_max(heatmap)

    # Safely normalize the heatmap between 0 and 1
    heatmap = tf.where(
        maximum_value > 0,
        heatmap / maximum_value,
        heatmap
    )

    return heatmap.numpy()


# =========================================================
# 9. OVERLAY GRAD-CAM ON ORIGINAL IMAGE
# =========================================================

def create_gradcam_overlay(image, heatmap):
    """
    Resize and overlay the Grad-CAM heatmap on the image.

    Typical color interpretation:
        Red/yellow = stronger influence
        Blue        = weaker influence
    """

    height, width = image.shape[:2]

    # Resize heatmap to original image dimensions
    resized_heatmap = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # Convert heatmap values from 0-1 to 0-255
    heatmap_uint8 = np.uint8(
        np.clip(resized_heatmap, 0, 1) * 255
    )

    # Convert grayscale heatmap into a colored map
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Combine the original image and heatmap
    overlay = cv2.addWeighted(
        image,
        0.60,
        colored_heatmap,
        0.40,
        0
    )

    return overlay


# =========================================================
# 10. CREATE INITIAL SEGMENTATION FROM GRAD-CAM
# =========================================================

def create_initial_segmentation_mask(image, heatmap):
    """
    Use the important Grad-CAM region as an initial estimate
    of the product foreground.

    This is weakly supervised segmentation because the
    classification model was trained using image-level labels,
    not pixel-level segmentation masks.
    """

    height, width = image.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # Convert from 0-1 float values to 0-255
    heatmap_uint8 = np.uint8(
        np.clip(resized_heatmap, 0, 1) * 255
    )

    # Otsu automatically calculates a threshold based on
    # heatmap intensity distribution.
    _, binary_mask = cv2.threshold(
        heatmap_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological closing fills small holes inside
    # the estimated foreground region.
    closing_kernel = np.ones(
        (9, 9),
        dtype=np.uint8
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2
    )

    # Morphological opening removes isolated noise.
    opening_kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1
    )

    return binary_mask


# =========================================================
# 11. REFINE SEGMENTATION USING OPENCV GRABCUT
# =========================================================

def refine_mask_with_grabcut(image, initial_mask):
    """
    Refine the initial Grad-CAM mask using OpenCV GrabCut.

    GrabCut labels:
        GC_BGD     = definite background
        GC_FGD     = definite foreground
        GC_PR_BGD  = probable background
        GC_PR_FGD  = probable foreground
    """

    height, width = image.shape[:2]

    # GrabCut mask starts with all pixels marked as
    # probable background.
    grabcut_mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    foreground_pixels = initial_mask > 0

    # Important Grad-CAM regions are probable foreground
    grabcut_mask[foreground_pixels] = cv2.GC_PR_FGD

    # Erode the mask to identify a smaller area that is
    # highly likely to be foreground.
    erosion_kernel = np.ones(
        (7, 7),
        dtype=np.uint8
    )

    sure_foreground = cv2.erode(
        initial_mask,
        erosion_kernel,
        iterations=1
    )

    grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

    # The outer image border is usually catalog background.
    border_size = max(
        2,
        int(min(height, width) * 0.02)
    )

    grabcut_mask[:border_size, :] = cv2.GC_BGD
    grabcut_mask[-border_size:, :] = cv2.GC_BGD
    grabcut_mask[:, :border_size] = cv2.GC_BGD
    grabcut_mask[:, -border_size:] = cv2.GC_BGD

    # GrabCut requires these temporary statistical models
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

        # Keep definite foreground and probable foreground
        final_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) |
            (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

    except cv2.error:
        # If GrabCut cannot refine a particular image,
        # safely use the initial Grad-CAM mask.
        final_mask = initial_mask.copy()

    # Keep only the largest connected foreground component.
    # This helps remove small noisy regions.
    final_mask = keep_largest_component(final_mask)

    # Smooth the final mask
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=2
    )

    return final_mask


# =========================================================
# 12. KEEP THE LARGEST FOREGROUND REGION
# =========================================================

def keep_largest_component(binary_mask):
    """
    Keep only the largest connected foreground component.

    Fashion catalog images normally contain one main product.
    Removing smaller components suppresses segmentation noise.
    """

    number_of_labels, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    # Label 0 is the background.
    if number_of_labels <= 1:
        return binary_mask

    component_areas = statistics[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_component_label = (
        1 + int(np.argmax(component_areas))
    )

    largest_mask = np.where(
        labels == largest_component_label,
        255,
        0
    ).astype(np.uint8)

    return largest_mask


# =========================================================
# 13. REMOVE THE BACKGROUND
# =========================================================

def create_transparent_product(image, mask):
    """
    Create a PNG image with a transparent background.

    The mask becomes the alpha channel:
        255 = visible product
        0   = transparent background
    """

    blue, green, red = cv2.split(image)

    transparent_image = cv2.merge(
        [blue, green, red, mask]
    )

    return transparent_image


# =========================================================
# 14. PLACE PRODUCT ON WHITE BACKGROUND
# =========================================================

def create_white_catalog_image(image, mask):
    """
    Put the segmented product on a uniform white background.
    """

    # Convert mask to a 0-1 floating-point alpha mask
    alpha = mask.astype(np.float32) / 255.0

    # Add the color-channel dimension:
    # H x W becomes H x W x 1
    alpha = np.expand_dims(alpha, axis=-1)

    # Create a completely white image
    white_background = np.full_like(
        image,
        255
    )

    # Alpha compositing:
    # foreground * alpha + background * (1 - alpha)
    catalog_image = (
        image.astype(np.float32) * alpha
        +
        white_background.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(
        catalog_image,
        0,
        255
    ).astype(np.uint8)


# =========================================================
# 15. CREATE PRODUCT BOUNDING BOX
# =========================================================

def get_product_bounding_box(mask):
    """
    Calculate a bounding box around the segmented product.

    Returns:
        (x, y, width, height)

    Returns None if no foreground was found.
    """

    foreground_points = cv2.findNonZero(mask)

    if foreground_points is None:
        return None

    x, y, width, height = cv2.boundingRect(
        foreground_points
    )

    return x, y, width, height


def draw_product_bounding_box(
    image,
    bounding_box,
    class_name,
    confidence
):
    """
    Draw the product location and prediction on the image.
    """

    detected_image = image.copy()

    if bounding_box is None:
        return detected_image

    x, y, width, height = bounding_box

    # Draw green bounding box
    cv2.rectangle(
        detected_image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    label = (
        f"{class_name}: "
        f"{confidence * 100:.2f}%"
    )

    # Calculate label size
    (text_width, text_height), baseline = (
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )
    )

    label_y = max(y, text_height + 10)

    # Draw filled rectangle behind text
    cv2.rectangle(
        detected_image,
        (x, label_y - text_height - 10),
        (x + text_width + 10, label_y),
        (0, 255, 0),
        -1
    )

    # Draw black label text
    cv2.putText(
        detected_image,
        label,
        (x + 5, label_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return detected_image


# =========================================================
# 16. END-TO-END DAY 8 CV PIPELINE
# =========================================================

def run_cv_pipeline(
    image_path,
    model,
    class_names
):
    """
    Run the complete Day 8 pipeline.

    Flow:
        Input image
            -> VGG16 preprocessing
            -> Product classification
            -> Grad-CAM heatmap
            -> Segmentation mask
            -> Background removal
            -> White catalog image
            -> Bounding-box localization
    """

    # Step 1: Load product image
    original_image = load_image(image_path)

    # Step 2: Prepare input using the exact Day 7 preprocessing
    model_input = prepare_image_for_vgg(
        original_image
    )

    # Step 3: Classify the product
    prediction = classify_product(
        model,
        model_input,
        class_names
    )

    # Step 4: Explain the predicted class with Grad-CAM
    heatmap = make_gradcam_heatmap(
        model_input=model_input,
        model=model,
        last_conv_layer_name=LAST_CONV_LAYER_NAME,
        class_index=prediction["class_index"]
    )

    # Step 5: Overlay Grad-CAM on the input image
    gradcam_overlay = create_gradcam_overlay(
        original_image,
        heatmap
    )

    # Step 6: Convert important model regions into
    # an initial segmentation mask
    initial_mask = create_initial_segmentation_mask(
        original_image,
        heatmap
    )

    # Step 7: Refine the segmentation using GrabCut
    final_mask = refine_mask_with_grabcut(
        original_image,
        initial_mask
    )

    # Step 8: Create transparent-background product
    transparent_product = create_transparent_product(
        original_image,
        final_mask
    )

    # Step 9: Put product on a uniform white background
    catalog_image = create_white_catalog_image(
        original_image,
        final_mask
    )

    # Step 10: Find product location from segmentation mask
    bounding_box = get_product_bounding_box(
        final_mask
    )

    # Step 11: Draw product bounding box and prediction
    detected_image = draw_product_bounding_box(
        original_image,
        bounding_box,
        prediction["class_name"],
        prediction["confidence"]
    )

    return {
        "original_image": original_image,
        "prediction": prediction,
        "heatmap": heatmap,
        "gradcam_overlay": gradcam_overlay,
        "initial_mask": initial_mask,
        "final_mask": final_mask,
        "transparent_product": transparent_product,
        "catalog_image": catalog_image,
        "bounding_box": bounding_box,
        "detected_image": detected_image
    }


# =========================================================
# 17. SAVE DAY 8 OUTPUTS
# =========================================================

def save_pipeline_outputs(
    result,
    image_path
):
    """
    Save images and prediction details generated by Day 8.
    """

    image_path = Path(image_path)
    image_name = image_path.stem

    # Create one output folder for this product image
    image_output_dir = OUTPUT_DIR / image_name

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    original_image = result["original_image"]

    # Save the original image for comparison
    cv2.imwrite(
        str(image_output_dir / "01_original.jpg"),
        original_image
    )

    # Save raw Grad-CAM as a grayscale heatmap
    resized_heatmap = cv2.resize(
        result["heatmap"],
        (
            original_image.shape[1],
            original_image.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )

    raw_heatmap = np.uint8(
        np.clip(resized_heatmap, 0, 1) * 255
    )

    cv2.imwrite(
        str(image_output_dir / "02_gradcam_heatmap.jpg"),
        raw_heatmap
    )

    # Save Grad-CAM overlay
    cv2.imwrite(
        str(image_output_dir / "03_gradcam_overlay.jpg"),
        result["gradcam_overlay"]
    )

    # Save initial segmentation mask
    cv2.imwrite(
        str(image_output_dir / "04_initial_mask.png"),
        result["initial_mask"]
    )

    # Save final refined segmentation mask
    cv2.imwrite(
        str(image_output_dir / "05_final_mask.png"),
        result["final_mask"]
    )

    # Save product with transparent background
    cv2.imwrite(
        str(image_output_dir / "06_transparent_product.png"),
        result["transparent_product"]
    )

    # Save clean catalog image
    cv2.imwrite(
        str(image_output_dir / "07_clean_catalog.jpg"),
        result["catalog_image"]
    )

    # Save product localization output
    cv2.imwrite(
        str(image_output_dir / "08_detected_product.jpg"),
        result["detected_image"]
    )

    bounding_box = result["bounding_box"]

    if bounding_box is None:
        bounding_box_data = None
    else:
        x, y, width, height = bounding_box

        bounding_box_data = {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "x_max": int(x + width),
            "y_max": int(y + height)
        }

    # Save prediction information in JSON format
    prediction_data = {
        "input_image": str(image_path),
        "predicted_class_index": (
            result["prediction"]["class_index"]
        ),
        "predicted_class_name": (
            result["prediction"]["class_name"]
        ),
        "confidence": round(
            result["prediction"]["confidence"],
            6
        ),
        "confidence_percentage": round(
            result["prediction"]["confidence"] * 100,
            2
        ),
        "bounding_box": bounding_box_data,
        "model": str(MODEL_PATH),
        "gradcam_layer": LAST_CONV_LAYER_NAME,
        "segmentation_method": (
            "Grad-CAM initialized OpenCV GrabCut"
        )
    }

    with open(
        image_output_dir / "prediction.json",
        mode="w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            prediction_data,
            json_file,
            indent=4
        )

    return image_output_dir


# =========================================================
# 18. MAIN PROGRAM
# =========================================================

def main():
    """
    Main function for running Day 8 from the terminal.
    """

    print("=" * 65)
    print("RETAILMIND AI - DAY 8 COMPUTER VISION PIPELINE")
    print("=" * 65)

    # Validate dataset and model paths
    validate_project_files()

    print("\nLoading class names...")

    class_names = load_class_names()

    print("Class names:", class_names)
    print("Number of classes:", len(class_names))

    print("\nLoading Day 7 VGG16 model...")
    print("Model path:", MODEL_PATH)

    model = load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # Validate that model output classes match the reconstructed
    # masterCategory class names.
    model_output_classes = int(model.output_shape[-1])

    if model_output_classes != len(class_names):
        raise ValueError(
            "\nClass-count mismatch detected.\n"
            f"Model output classes: {model_output_classes}\n"
            f"Reconstructed classes: {len(class_names)}\n\n"
            "The current styles.csv or images folder may differ "
            "from the dataset used to train Day 7."
        )

    # User can optionally provide an image:
    #
    # python src/Day8_segmentation_gradcam.py image_path.jpg
    if len(sys.argv) > 1:
        input_image_path = Path(sys.argv[1])
    else:
        input_image_path = get_default_image_path()

    print("\nInput image:", input_image_path)
    print("Running classification...")
    print("Generating Grad-CAM...")
    print("Segmenting product...")
    print("Creating clean catalog image...")

    result = run_cv_pipeline(
        image_path=input_image_path,
        model=model,
        class_names=class_names
    )

    output_folder = save_pipeline_outputs(
        result=result,
        image_path=input_image_path
    )

    prediction = result["prediction"]

    print("\n" + "=" * 65)
    print("DAY 8 RESULT")
    print("=" * 65)

    print(
        "Predicted category:",
        prediction["class_name"]
    )

    print(
        "Class index:",
        prediction["class_index"]
    )

    print(
        "Confidence:",
        f"{prediction['confidence'] * 100:.2f}%"
    )

    if result["bounding_box"] is not None:
        x, y, width, height = result["bounding_box"]

        print(
            "Bounding box:",
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        )
    else:
        print("Bounding box: Product region not found")

    print("\nOutputs saved inside:")
    print(output_folder)

    print("\nGenerated files:")

    for output_file in sorted(output_folder.iterdir()):
        print(" -", output_file.name)

    print("\nDay 8 pipeline completed successfully.")


if __name__ == "__main__":
    main()