"""
# Simpler scale based (1-255) normalization for all 3 color channels
Manual Implementation of DB Text Detection (Replacing cv2.dnn_TextDetectionModel_DB)

This module provides a complete replacement for OpenCV's TextDetectionModel_DB,
giving full control over the detection pipeline using ONNX Runtime.

Author: Computer Vision Engineer
Purpose: Replace OpenCV's black-box implementation with explicit, debuggable code
"""

import numpy as np
import cv2
import onnxruntime as ort
import pyclipper
from typing import Tuple, List, Dict, Any


# ============================================
# STEP 1: Model Loading (replaces cv2.dnn_TextDetectionModel_DB)
# ============================================

def load_db_model_manual(model_path: str) -> ort.InferenceSession:
    """
    Load ONNX model using ONNX Runtime.

    Replaces: cv2.dnn_TextDetectionModel_DB(modelPath)

    Args:
        model_path: Path to the ONNX model file (e.g., "DB_TD500_resnet50.onnx")

    Returns:
        ort.InferenceSession: Loaded ONNX model session ready for inference

    Notes:
        - Uses CPU execution provider by default
        - For GPU: add providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    """
    try:
        # Create ONNX Runtime session with optimizations
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model (CPU by default)
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CUDAExecutionProvider']
        )

        print(f"[OK] Model loaded successfully: {model_path}")
        print(f"  - Input name: {session.get_inputs()[0].name}")
        print(f"  - Input shape: {session.get_inputs()[0].shape}")
        print(f"  - Output name: {session.get_outputs()[0].name}")
        print(f"  - Output shape: {session.get_outputs()[0].shape}")

        return session

    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")


# ============================================
# STEP 2: Preprocessing (replaces setInputParams)
# ============================================

def preprocess_image_manual(
    image: np.ndarray,
    input_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    scale: float,
    swap_rb: bool
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Manual preprocessing matching OpenCV's internal logic.

    Replaces: setInputParams(scale, inputSize, mean, swapRB)

    This function replicates what OpenCV does internally when you call setInputParams
    and then detect(). The preprocessing pipeline:

    1. Resize image to model input size
    2. Swap R and B channels if swap_rb=True (BGR -> RGB)
    3. Subtract mean values from each channel
    4. Multiply by scale factor (typically 1/255 for normalization)
    5. Convert from HWC (Height, Width, Channel) to NCHW (Batch, Channel, Height, Width)
    6. Ensure float32 dtype for ONNX inference

    Args:
        image: Input image in BGR format, shape (H, W, 3)
        input_size: Target size as (width, height), e.g., (320, 320)
        mean: Mean values for each channel (B_mean, G_mean, R_mean)
        scale: Scaling factor, typically 1.0/255.0
        swap_rb: If True, swap Red and Blue channels (BGR -> RGB)

    Returns:
        preprocessed_blob: numpy array of shape (1, 3, H, W) ready for inference
        original_size: Original image size as (height, width) for later scaling
    """
    # Store original dimensions for coordinate scaling later
    original_size = (image.shape[0], image.shape[1])  # (height, width)

    # STEP 1: Resize image to model input size
    # OpenCV's dnn module uses INTER_LINEAR (bilinear interpolation) by default
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    print(f"  [Preprocess] Resized: {original_size} -> {input_size}")

    # STEP 2: Convert to float32 for numerical operations
    img_float = resized.astype(np.float32)

    # STEP 3: Swap R and B channels if required (BGR -> RGB)
    if swap_rb:
        # OpenCV loads images as BGR, most models expect RGB
        img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
        print(f"  [Preprocess] Color swap: BGR -> RGB")

    # STEP 4: Subtract mean values
    # Mean is typically computed from training dataset (ImageNet statistics)
    # Format: (mean_channel_0, mean_channel_1, mean_channel_2)
    # After swap_rb: (R_mean, G_mean, B_mean) if swap_rb=True
    img_float -= np.array(mean, dtype=np.float32)
    print(f"  [Preprocess] Mean subtraction: {mean}")

    # STEP 5: Apply scaling (typically 1/255 to normalize [0, 255] -> [0, 1])
    img_float *= scale
    print(f"  [Preprocess] Scaling: x{scale}")

    # STEP 6: Convert from HWC to NCHW format (OpenCV dnn format)
    # HWC: (Height, Width, Channels) - standard image format
    # NCHW: (Batch, Channels, Height, Width) - neural network format
    blob = img_float.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    blob = np.expand_dims(blob, axis=0)   # (C, H, W) -> (1, C, H, W)

    print(f"  [Preprocess] Blob shape: {blob.shape} (NCHW format)")

    return blob, original_size


# ============================================
# STEP 3: Inference
# ============================================

def run_inference_manual(
    onnx_session: ort.InferenceSession,
    input_blob: np.ndarray
) -> np.ndarray:
    """
    Run ONNX model inference.

    Args:
        onnx_session: Loaded ONNX Runtime inference session
        input_blob: Preprocessed input tensor, shape (1, 3, H, W)

    Returns:
        output: Model output tensor (typically probability/binary map)
                Shape is usually (1, 1, H, W) for DB models

    Notes:
        - DB models output a probability map indicating text regions
        - Some models may output multiple maps (binary, threshold, etc.)
    """
    # Get input name from the model
    input_name = onnx_session.get_inputs()[0].name

    # Run inference
    print(f"  [Inference] Running model...")
    outputs = onnx_session.run(None, {input_name: input_blob})

    # DB models typically output a single probability map
    # Shape: (1, 1, H, W) where values indicate text probability
    output = outputs[0]

    print(f"  [Inference] Output shape: {output.shape}")
    print(f"  [Inference] Output range: [{output.min():.3f}, {output.max():.3f}]")

    return output


# ============================================
# STEP 4: Post-processing (replaces detect() internals)
# ============================================

def unclip_polygon(contour: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray:
    """
    Apply polygon offset using pyclipper (same as OpenCV's internal implementation).

    CRITICAL UNDERSTANDING:
    - The DB model outputs a slightly SHRUNKEN text segmentation (conservative detection)
    - We need to EXPAND it back to cover the full text region
    - This is called "unclip" - we "un-clip" the shrunk segmentation to full text bounds

    Args:
        contour: Polygon contour as numpy array of shape (N, 1, 2) or (N, 2)
        unclip_ratio: Ratio for offsetting. Default 1.5 (expand by 50%)

    Returns:
        Expanded polygon as numpy array (covers full text region)
    """
    # Reshape contour to (N, 2) if it's (N, 1, 2)
    if len(contour.shape) == 3:
        contour = contour.reshape(-1, 2)

    # Calculate polygon area and perimeter
    area = cv2.contourArea(contour.astype(np.float32))
    perimeter = cv2.arcLength(contour.astype(np.float32), True)

    if perimeter == 0:
        return contour

    # CRITICAL: The unclip_ratio is how DB algorithm works
    # - The model outputs SHRUNK text regions (for better precision)
    # - We use unclip to EXPAND them back to full text bounds
    # - Standard DB unclip_ratio = 1.5 (expand by 50%)
    #
    # This is NOT optional - it's part of the DB algorithm design!
    # The dilation we did earlier connects pixels, but unclip gives proper bounds

    # Use the standard DB unclip ratio
    # This matches what OpenCV uses internally
    distance = area * (unclip_ratio - 1.0) / perimeter  # 1.5 gives 50% expansion

    # Use pyclipper to offset the polygon
    pco = pyclipper.PyclipperOffset()

    # Convert contour to list of tuples (pyclipper format)
    contour_list = contour.astype(np.int32).tolist()

    # Add path to clipper
    pco.AddPath(contour_list, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    # Execute offset with POSITIVE distance for expansion
    # This expands the shrunk segmentation to the full text box
    result = pco.Execute(distance)  # Positive = expand

    # Handle the result
    if len(result) == 0:
        # If expansion failed, return original contour
        return contour

    # Take the first (and usually only) polygon
    expanded_polygon = np.array(result[0], dtype=np.float32)

    return expanded_polygon


def postprocess_db_output_manual(
    model_output: np.ndarray,
    binary_threshold: float,
    polygon_threshold: float,
    original_size: Tuple[int, int],
    input_size: Tuple[int, int]
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract text boxes and confidences from DB model output.

    Replaces: The internal post-processing of textDetectorDB.detect()

    This function implements the complete DB post-processing pipeline:
    1. Extract probability map from model output
    2. Apply binary threshold to create binary mask
    3. Find contours in the binary mask
    4. Approximate contours as polygons
    5. Calculate confidence score for each polygon
    6. Filter polygons by polygon_threshold
    7. Scale polygon coordinates back to original image size

    Args:
        model_output: Raw output from ONNX model, shape (1, 1, H, W)
        binary_threshold: Threshold for binarizing probability map (e.g., 0.3)
                         Pixels with prob > binary_threshold are considered text
        polygon_threshold: Minimum confidence for keeping detections (e.g., 0.5)
                          Polygons with score < polygon_threshold are discarded
        original_size: Original image dimensions (height, width)
        input_size: Model input dimensions (width, height)

    Returns:
        boxes: List of detected text polygons, each as numpy array of shape (N, 2)
               Format: [[x1, y1], [x2, y2], ..., [xN, yN]]
        confidences: List of confidence scores for each box (0.0 to 1.0)

    Notes:
        - OpenCV's TextDetectionModel_DB returns polygons with 4+ vertices
        - Confidence is computed as mean probability within the polygon region
    """
    print(f"  [Postprocess] Binary threshold: {binary_threshold}")
    print(f"  [Postprocess] Polygon threshold: {polygon_threshold}")

    # STEP 1: Extract probability map (handle different output shapes)
    # Some models output (1, 1, H, W), others output (H, W) or (1, H, W)
    if len(model_output.shape) == 4:
        # Shape: (1, 1, H, W) -> (H, W)
        prob_map = model_output[0, 0, :, :]
    elif len(model_output.shape) == 3:
        # Shape: (1, H, W) -> (H, W)
        prob_map = model_output[0, :, :]
    elif len(model_output.shape) == 2:
        # Shape: (H, W) - already correct
        prob_map = model_output
    else:
        raise ValueError(f"Unexpected model output shape: {model_output.shape}")

    print(f"  [Postprocess] Probability map shape: {prob_map.shape}")

    # STEP 2: Apply binary threshold to create binary mask
    # CRITICAL: OpenCV uses TWO-STAGE thresholding (this is the key!)
    # Stage 1: Low threshold (0.3) to find all text regions
    # Stage 2: High threshold (0.7) to get tight text-only pixels
    #
    # We create TWO binary maps:
    # - Low threshold map: For finding contours (includes some background)
    # - High threshold map: For final box extraction (text only)

    # CRITICAL FIX: Use the binary_threshold map directly (0.3)
    # OpenCV uses the low threshold for finding contours to get full text extent
    # The high threshold approach was making boxes too small vertically
    binary_map = (prob_map > binary_threshold).astype(np.uint8) * 255

    print(f"  [Postprocess] Binary threshold: {binary_threshold}")
    print(f"  [Postprocess] Binary map: {binary_map.shape}, "
          f"text pixels: {np.sum(binary_map > 0)}")

    # STEP 2.5: Apply morphological CLOSE to connect text and remove noise
    # CLOSING = dilation followed by erosion
    # - Connects nearby characters into words
    # - Fills small holes in text
    # - Returns to approximately original size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=1)

    # This is our final binary map for contour detection
    print(f"  [Postprocess] After morphological closing: text pixels: {np.sum(binary_map > 0)}")

    # STEP 3: Find contours in the binary map
    # Use RETR_EXTERNAL to get only outer-most contours
    # This prevents detecting nested contours (like text inside background boxes)
    # cv2.CHAIN_APPROX_SIMPLE: compress horizontal, vertical, and diagonal segments
    contours, hierarchy = cv2.findContours(
        binary_map,
        cv2.RETR_EXTERNAL,  # Get only external contours (no nested)
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"  [Postprocess] Found {len(contours)} contours")


    # Sort contours by area (smallest first)
    # This prioritizes text regions over background boxes
    contours = sorted(contours, key=cv2.contourArea)

    # Lists to store valid boxes and their confidences
    boxes = []
    confidences = []
    processed_regions = []  # Track regions we've already detected

    # Calculate scaling factors to map from model input size to original image size
    scale_x = original_size[1] / input_size[0]  # width scaling
    scale_y = original_size[0] / input_size[1]  # height scaling

    # STEP 4: Process each contour
    for contour in contours:
        # Need at least 4 points for a valid polygon
        if len(contour) < 4:
            continue

        # Calculate polygon area to filter out tiny regions
        area = cv2.contourArea(contour)
        if area < 10:  # Minimum area threshold (in model input space)
            continue

        # STEP 5: Calculate confidence score FIRST (before expanding)
        # Method: compute mean probability within the original contour region
        # This ensures we don't go out of bounds when the polygon expands

        # Create mask for the ORIGINAL contour
        mask = np.zeros_like(prob_map, dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 255)

        # Calculate mean probability in the contour region
        polygon_region = prob_map[mask > 0]
        if len(polygon_region) == 0:
            confidence = 0.0
            max_prob = 0.0
        else:
            confidence = float(np.mean(polygon_region))
            max_prob = float(np.max(polygon_region))

        # STEP 6: Filter by polygon threshold
        if confidence < polygon_threshold:
            continue  # Discard low-confidence detections

        # CRITICAL FILTER: Distinguish text from background boxes
        # Strategy: Use ratio of high-prob pixels to total pixels
        # - Real text: Most pixels have high probability (tight distribution)
        # - Background box: Mix of high (text) and medium (background) pixels

        # Count pixels above different thresholds
        high_conf_pixels = np.sum(polygon_region > 0.65)  # Lower threshold to 0.65
        total_pixels = len(polygon_region)

        if total_pixels == 0:
            continue

        high_conf_ratio = high_conf_pixels / total_pixels

        # Real text regions should have >40% pixels with confidence >0.65
        # Background boxes have <30% because most is low-confidence background
        # Lowered from 0.6 to 0.4 to avoid rejecting valid text
        if high_conf_ratio < 0.4:
            continue  # Reject regions with too much low-confidence area (backgrounds)

        # CRITICAL: Apply pyclipper polygon expansion (unclip)
        # This is ESSENTIAL to the DB algorithm and what OpenCV uses internally
        # The model outputs SHRUNK text regions (conservative), we expand them back
        # Using unclip_ratio=3.15 to closely match OpenCV's box sizes
        try:
            expanded_contour = unclip_polygon(contour, unclip_ratio=3.15)
        except:
            # If unclipping fails, use original contour
            expanded_contour = contour

        # Get bounding box from the expanded contour
        rect = cv2.minAreaRect(expanded_contour)
        polygon = cv2.boxPoints(rect)

        # boxPoints returns (4, 2) array of corner points
        polygon = polygon.astype(np.float32)

        # Clip polygon coordinates to stay within bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, input_size[0] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, input_size[1] - 1)

        # STEP 7: Scale polygon coordinates back to original image size
        # Coordinates are currently in model input space (e.g., 320×320)
        # We need to scale them to original image space
        polygon_scaled = polygon.astype(np.float32)
        polygon_scaled[:, 0] *= scale_x  # Scale x coordinates
        polygon_scaled[:, 1] *= scale_y  # Scale y coordinates

        # Convert back to integers for final output
        polygon_scaled = polygon_scaled.astype(np.int32)

        # Add to results
        boxes.append(polygon_scaled)
        confidences.append(confidence)

        # Mark this region as processed (store unscaled polygon)
        processed_regions.append(polygon)

    print(f"  [Postprocess] Filtered to {len(boxes)} valid detections "
          f"(above threshold {polygon_threshold})")

    return boxes, confidences


# ============================================
# STEP 5: Complete Pipeline (Full Replacement)
# ============================================

def detect_text_manual(
    image: np.ndarray,
    onnx_session: ort.InferenceSession,
    config: Dict[str, Any]
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Complete replacement for textDetectorDB.detect(image).

    This function combines all steps to provide the same interface as OpenCV's
    TextDetectionModel_DB.detect() method.

    Pipeline:
    1. Preprocess image (resize, normalize, create blob)
    2. Run inference through ONNX model
    3. Post-process outputs (threshold, find contours, filter, scale)
    4. Return boxes and confidences in OpenCV-compatible format

    Args:
        image: Input image in BGR format, shape (H, W, 3)
        onnx_session: Loaded ONNX Runtime inference session
        config: Configuration dictionary containing:
            - 'binary_threshold': Threshold for binarization (e.g., 0.3)
            - 'polygon_threshold': Minimum confidence for detections (e.g., 0.5)
            - 'input_size': Model input size as (width, height), e.g., (320, 320)
            - 'mean': Mean values (B, G, R) or (R, G, B) after swap
            - 'scale': Scaling factor, typically 1.0/255
            - 'swap_rb': Boolean, whether to swap R and B channels

    Returns:
        boxes: List of detected text boxes as numpy arrays of shape (N, 2)
               Each box is a polygon: [[x1, y1], [x2, y2], ..., [xN, yN]]
        confidences: List of confidence scores (0.0 to 1.0) for each box

    Example:
        >>> session = load_db_model_manual("DB_TD500_resnet50.onnx")
        >>> config = {
        ...     'binary_threshold': 0.3,
        ...     'polygon_threshold': 0.5,
        ...     'input_size': (320, 320),
        ...     'mean': (122.67891434, 116.66876762, 104.00698793),
        ...     'scale': 1.0/255,
        ...     'swap_rb': True
        ... }
        >>> image = cv2.imread("test.jpg")
        >>> boxes, confidences = detect_text_manual(image, session, config)
        >>> print(f"Detected {len(boxes)} text regions")
    """
    print(f"\n{'='*60}")
    print(f"MANUAL DB TEXT DETECTION PIPELINE")
    print(f"{'='*60}")

    # Extract configuration parameters
    binary_threshold = config['binary_threshold']
    polygon_threshold = config['polygon_threshold']
    input_size = config['input_size']
    mean = config['mean']
    scale = config['scale']
    swap_rb = config['swap_rb']

    # STEP 1: Preprocess image
    print(f"\n[1/3] PREPROCESSING")
    input_blob, original_size = preprocess_image_manual(
        image, input_size, mean, scale, swap_rb
    )

    # STEP 2: Run inference
    print(f"\n[2/3] INFERENCE")
    model_output = run_inference_manual(onnx_session, input_blob)

    # STEP 3: Post-process to get boxes and confidences
    print(f"\n[3/3] POST-PROCESSING")
    boxes, confidences = postprocess_db_output_manual(
        model_output,
        binary_threshold,
        polygon_threshold,
        original_size,
        input_size
    )

    print(f"\n{'='*60}")
    print(f"DETECTION COMPLETE: {len(boxes)} text regions found")
    print(f"{'='*60}\n")

    return boxes, confidences


# ============================================
# HELPER FUNCTION: Visualize Results
# ============================================

def visualize_detections(
    image: np.ndarray,
    boxes: List[np.ndarray],
    confidences: List[float],
    output_path: str = None
) -> np.ndarray:
    """
    Draw detected text boxes on image for visualization.

    Args:
        image: Input image (BGR format)
        boxes: List of polygon boxes from detect_text_manual()
        confidences: List of confidence scores
        output_path: If provided, save visualization to this path

    Returns:
        vis_image: Image with drawn boxes
    """
    vis_image = image.copy()

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        # Draw polygon in BLUE (manual detection color)
        cv2.polylines(vis_image, [box], isClosed=True,
                     color=(255, 0, 0), thickness=2)

        # Draw confidence score in BLUE
        x, y = box[0]
        text = f"{conf:.2f}"
        cv2.putText(vis_image, text, (int(x), int(y) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")

    return vis_image


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    """
    Example showing how to replace OpenCV's TextDetectionModel_DB
    with the manual implementation.
    """

    print("\n" + "="*70)
    print("DB TEXT DETECTION - MANUAL IMPLEMENTATION")
    print("Replacing cv2.dnn_TextDetectionModel_DB with ONNX Runtime")
    print("="*70 + "\n")

    # ========================================
    # CONFIGURATION
    # ========================================

    # Model path (replace with your actual path)
    model_path = "/home/manas/Downloads/em-multilingual Nov5/EM-Multilingual/model/DB_TD500_resnet50.onnx"

    # Detection configuration (replaces setBinaryThreshold, setPolygonThreshold, setInputParams)
    config = {
        'binary_threshold': 0.3,     # setBinaryThreshold(0.3)
        'polygon_threshold': 0.5,    # setPolygonThreshold(0.5)
        'input_size': (320, 320),    # setInputParams(..., inputSize=(320, 320), ...)
        'mean': (122.67891434, 116.66876762, 104.00698793),  # setInputParams(..., mean=..., ...)
        'scale': 1.0/255,            # setInputParams(scale=1.0/255, ...)
        'swap_rb': True              # setInputParams(..., swapRB=True)
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # ========================================
    # OLD WAY (OpenCV's black-box implementation)
    # ========================================
    """
    # This is what we're replacing:
    textDetector = cv2.dnn_TextDetectionModel_DB(model_path)
    textDetector.setBinaryThreshold(0.3)
    textDetector.setPolygonThreshold(0.5)
    textDetector.setInputParams(1.0/255, (320, 320),
                                (122.67891434, 116.66876762, 104.00698793),
                                True)
    boxes, confidences = textDetector.detect(image)
    """

    # ========================================
    # NEW WAY (Manual implementation with full control)
    # ========================================

    # STEP 1: Load model (do this once at startup)
    print("STEP 1: Loading ONNX model...")
    try:
        session = load_db_model_manual(model_path)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Please provide the correct path to DB_TD500_resnet50.onnx")
        print("You can download it from: https://github.com/opencv/opencv_zoo")
        exit(1)

    # STEP 2: Load test image
    print("\nSTEP 2: Loading test image...")
    image_path = "/home/manas/Downloads/em-multilingual Nov5/EM-Multilingual/data/frame_000030_original.jpg"  # Replace with your test image

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        print(f"[OK] Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Please provide a valid test image path")
        print("Creating a dummy image for demonstration...")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # STEP 3: Detect text (replaces textDetector.detect(image))
    print("\nSTEP 3: Detecting text...")
    boxes, confidences = detect_text_manual(image, session, config)

    # STEP 4: Display results
    print("\nRESULTS:")
    print(f"  Total detections: {len(boxes)}")

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        print(f"  Box {i+1}: {len(box)} vertices, confidence: {conf:.3f}")
        print(f"    Coordinates: {box[:2]}...")  # Show first 2 vertices

    # STEP 5: Visualize (optional)
    if len(boxes) > 0:
        print("\nSTEP 4: Creating visualization...")
        vis_image = visualize_detections(image, boxes, confidences,
                                        output_path="detection_result.jpg")
        print("[OK] Detection complete!")

    print("\n" + "="*70)
    print("COMPARISON: OpenCV vs Manual Implementation")
    print("="*70)
    print("""
    OLD (OpenCV):                    NEW (Manual):
    ─────────────                    ─────────────

    textDetector = cv2.dnn_          session = load_db_model_manual(
        TextDetectionModel_DB(           model_path)
        model_path)
                                     config = {
    textDetector.setBinary               'binary_threshold': 0.3,
        Threshold(0.3)                   'polygon_threshold': 0.5,
    textDetector.setPolygon              'input_size': (320, 320),
        Threshold(0.5)                   'mean': (...),
    textDetector.setInputParams(         'scale': 1.0/255,
        1.0/255, (320, 320),             'swap_rb': True
        mean, True)                  }

    boxes, confidences =             boxes, confidences =
        textDetector.detect(image)       detect_text_manual(
                                             image, session, config)

    [OK] Same output format
    [OK] Same results (boxes + confidences)
    [OK] Full control over pipeline
    [OK] Easy to debug and modify
    """)
    print("="*70 + "\n")


















# """
# Standard Deviation based normalization for all 3 color channels
# Manual Implementation of DB Text Detection (Replacing cv2.dnn_TextDetectionModel_DB)

# This module provides a complete replacement for OpenCV's TextDetectionModel_DB,
# giving full control over the detection pipeline using ONNX Runtime.

# Author: Computer Vision Engineer
# Purpose: Replace OpenCV's black-box implementation with explicit, debuggable code
# """

# import numpy as np
# import cv2
# import onnxruntime as ort
# import pyclipper
# from typing import Tuple, List, Dict, Any


# # ============================================
# # STEP 1: Model Loading (replaces cv2.dnn_TextDetectionModel_DB)
# # ============================================

# def load_db_model_manual(model_path: str) -> ort.InferenceSession:
#     """
#     Load ONNX model using ONNX Runtime.

#     Replaces: cv2.dnn_TextDetectionModel_DB(modelPath)

#     Args:
#         model_path: Path to the ONNX model file (e.g., "DB_TD500_resnet50.onnx")

#     Returns:
#         ort.InferenceSession: Loaded ONNX model session ready for inference

#     Notes:
#         - Uses CPU execution provider by default
#         - For GPU: add providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
#     """
#     try:
#         # Create ONNX Runtime session with optimizations
#         session_options = ort.SessionOptions()
#         session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

#         # Load the model (CPU by default)
#         session = ort.InferenceSession(
#             model_path,
#             sess_options=session_options,
#             providers=['CPUExecutionProvider']
#         )

#         print(f"[OK] Model loaded successfully: {model_path}")
#         print(f"  - Input name: {session.get_inputs()[0].name}")
#         print(f"  - Input shape: {session.get_inputs()[0].shape}")
#         print(f"  - Output name: {session.get_outputs()[0].name}")
#         print(f"  - Output shape: {session.get_outputs()[0].shape}")

#         return session

#     except Exception as e:
#         raise RuntimeError(f"Failed to load ONNX model: {e}")


# # ============================================
# # STEP 2: Preprocessing (replaces setInputParams)
# # ============================================

# def preprocess_image_manual(
#     image: np.ndarray,
#     input_size: Tuple[int, int],
#     mean: Tuple[float, float, float],
#     std: Tuple[float, float, float],
#     swap_rb: bool
# ) -> Tuple[np.ndarray, Tuple[int, int]]:
#     """
#     Manual preprocessing matching OpenCV's internal logic.

#     Replaces: setInputParams(scale, inputSize, mean, swapRB)

#     This function replicates what OpenCV does internally when you call setInputParams
#     and then detect(). The preprocessing pipeline:

#     1. Resize image to model input size
#     2. Swap R and B channels if swap_rb=True (BGR -> RGB)
#     3. Subtract mean values from each channel
#     4. Divide by standard deviation for per-channel normalization (CRITICAL)
#     5. Convert from HWC (Height, Width, Channel) to NCHW (Batch, Channel, Height, Width)
#     6. Ensure float32 dtype for ONNX inference

#     Args:
#         image: Input image in BGR format, shape (H, W, 3)
#         input_size: Target size as (width, height), e.g., (736, 736)
#         mean: Mean values for each channel (B_mean, G_mean, R_mean)
#         std: Standard deviation values for each channel for normalization
#         swap_rb: If True, swap Red and Blue channels (BGR -> RGB)

#     Returns:
#         preprocessed_blob: numpy array of shape (1, 3, H, W) ready for inference
#         original_size: Original image size as (height, width) for later scaling
#     """
#     # Store original dimensions for coordinate scaling later
#     original_size = (image.shape[0], image.shape[1])  # (height, width)

#     # STEP 1: Resize image to model input size
#     # OpenCV's dnn module uses INTER_LINEAR (bilinear interpolation) by default
#     resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
#     print(f"  [Preprocess] Resized: {original_size} -> {input_size}")

#     # STEP 2: Convert to float32 for numerical operations
#     img_float = resized.astype(np.float32)

#     # STEP 3: Swap R and B channels if required (BGR -> RGB)
#     if swap_rb:
#         # OpenCV loads images as BGR, most models expect RGB
#         img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
#         print(f"  [Preprocess] Color swap: BGR -> RGB")

#     # STEP 4: Subtract mean values
#     # Mean is typically computed from training dataset (ImageNet statistics)
#     # Format: (mean_channel_0, mean_channel_1, mean_channel_2)
#     # After swap_rb: (R_mean, G_mean, B_mean) if swap_rb=True
#     img_float -= np.array(mean, dtype=np.float32)
#     print(f"  [Preprocess] Mean subtraction: {mean}")

#     # STEP 5: Normalize by standard deviation (per-channel)
#     # This is more accurate than simple scaling
#     img_float /= np.array(std, dtype=np.float32)
#     print(f"  [Preprocess] Std deviation normalization: {std}")

#     # STEP 6: Convert from HWC to NCHW format (OpenCV dnn format)
#     # HWC: (Height, Width, Channels) - standard image format
#     # NCHW: (Batch, Channels, Height, Width) - neural network format
#     blob = img_float.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
#     blob = np.expand_dims(blob, axis=0)   # (C, H, W) -> (1, C, H, W)

#     print(f"  [Preprocess] Blob shape: {blob.shape} (NCHW format)")

#     return blob, original_size


# # ============================================
# # STEP 3: Inference
# # ============================================

# def run_inference_manual(
#     onnx_session: ort.InferenceSession,
#     input_blob: np.ndarray
# ) -> np.ndarray:
#     """
#     Run ONNX model inference.

#     Args:
#         onnx_session: Loaded ONNX Runtime inference session
#         input_blob: Preprocessed input tensor, shape (1, 3, H, W)

#     Returns:
#         output: Model output tensor (typically probability/binary map)
#                 Shape is usually (1, 1, H, W) for DB models

#     Notes:
#         - DB models output a probability map indicating text regions
#         - Some models may output multiple maps (binary, threshold, etc.)
#     """
#     # Get input name from the model
#     input_name = onnx_session.get_inputs()[0].name

#     # Run inference
#     print(f"  [Inference] Running model...")
#     outputs = onnx_session.run(None, {input_name: input_blob})

#     # DB models typically output a single probability map
#     # Shape: (1, 1, H, W) where values indicate text probability
#     output = outputs[0]

#     print(f"  [Inference] Output shape: {output.shape}")
#     print(f"  [Inference] Output range: [{output.min():.3f}, {output.max():.3f}]")

#     return output


# # ============================================
# # STEP 4: Post-processing (replaces detect() internals)
# # ============================================

# def unclip_polygon(contour: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray:
#     """
#     Apply polygon offset using pyclipper (same as OpenCV's internal implementation).

#     CRITICAL UNDERSTANDING:
#     - The DB model outputs a slightly SHRUNKEN text segmentation (conservative detection)
#     - We need to EXPAND it back to cover the full text region
#     - This is called "unclip" - we "un-clip" the shrunk segmentation to full text bounds

#     Args:
#         contour: Polygon contour as numpy array of shape (N, 1, 2) or (N, 2)
#         unclip_ratio: Ratio for offsetting. Default 1.5 (expand by 50%)

#     Returns:
#         Expanded polygon as numpy array (covers full text region)
#     """
#     # Reshape contour to (N, 2) if it's (N, 1, 2)
#     if len(contour.shape) == 3:
#         contour = contour.reshape(-1, 2)

#     # Calculate polygon area and perimeter
#     area = cv2.contourArea(contour.astype(np.float32))
#     perimeter = cv2.arcLength(contour.astype(np.float32), True)

#     if perimeter == 0:
#         return contour

#     # CRITICAL: The unclip_ratio is how DB algorithm works
#     # - The model outputs SHRUNK text regions (for better precision)
#     # - We use unclip to EXPAND them back to full text bounds
#     # - Standard DB unclip_ratio = 1.5 (expand by 50%)
#     #
#     # This is NOT optional - it's part of the DB algorithm design!
#     # The dilation we did earlier connects pixels, but unclip gives proper bounds

#     # Use the standard DB unclip ratio
#     # This matches what OpenCV uses internally
#     distance = area * (unclip_ratio - 1.0) / perimeter  # 1.5 gives 50% expansion

#     # Use pyclipper to offset the polygon
#     pco = pyclipper.PyclipperOffset()

#     # Convert contour to list of tuples (pyclipper format)
#     contour_list = contour.astype(np.int32).tolist()

#     # Add path to clipper
#     pco.AddPath(contour_list, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

#     # Execute offset with POSITIVE distance for expansion
#     # This expands the shrunk segmentation to the full text box
#     result = pco.Execute(distance)  # Positive = expand

#     # Handle the result
#     if len(result) == 0:
#         # If expansion failed, return original contour
#         return contour

#     # Take the first (and usually only) polygon
#     expanded_polygon = np.array(result[0], dtype=np.float32)

#     return expanded_polygon


# def postprocess_db_output_manual(
#     model_output: np.ndarray,
#     binary_threshold: float,
#     polygon_threshold: float,
#     original_size: Tuple[int, int],
#     input_size: Tuple[int, int],
#     delta_width_ratio: float = 0.05,
#     delta_height_ratio: float = 0.10,
#     min_height_padding_pixels: int = 5
# ) -> Tuple[List[np.ndarray], List[float]]:
#     """
#     Extract text boxes and confidences from DB model output.

#     Replaces: The internal post-processing of textDetectorDB.detect()

#     This function implements the complete DB post-processing pipeline:
#     1. Extract probability map from model output
#     2. Apply binary threshold to create binary mask
#     3. Find contours in the binary mask
#     4. Approximate contours as polygons
#     5. Calculate confidence score for each polygon
#     6. Filter polygons by polygon_threshold
#     7. Scale polygon coordinates back to original image size

#     Args:
#         model_output: Raw output from ONNX model, shape (1, 1, H, W)
#         binary_threshold: Threshold for binarizing probability map (e.g., 0.3)
#                          Pixels with prob > binary_threshold are considered text
#         polygon_threshold: Minimum confidence for keeping detections (e.g., 0.5)
#                           Polygons with score < polygon_threshold are discarded
#         original_size: Original image dimensions (height, width)
#         input_size: Model input dimensions (width, height)
#         delta_width_ratio: Padding ratio for width (default 0.05 = 5%)
#         delta_height_ratio: Padding ratio for height (default 0.10 = 10%)
#         min_height_padding_pixels: Minimum height padding in pixels (default 5)

#     Returns:
#         boxes: List of detected text polygons, each as numpy array of shape (N, 2)
#                Format: [[x1, y1], [x2, y2], ..., [xN, yN]]
#         confidences: List of confidence scores for each box (0.0 to 1.0)

#     Notes:
#         - OpenCV's TextDetectionModel_DB returns polygons with 4+ vertices
#         - Confidence is computed as mean probability within the polygon region
#     """
#     print(f"  [Postprocess] Binary threshold: {binary_threshold}")
#     print(f"  [Postprocess] Polygon threshold: {polygon_threshold}")

#     # STEP 1: Extract probability map (handle different output shapes)
#     # Some models output (1, 1, H, W), others output (H, W) or (1, H, W)
#     if len(model_output.shape) == 4:
#         # Shape: (1, 1, H, W) -> (H, W)
#         prob_map = model_output[0, 0, :, :]
#     elif len(model_output.shape) == 3:
#         # Shape: (1, H, W) -> (H, W)
#         prob_map = model_output[0, :, :]
#     elif len(model_output.shape) == 2:
#         # Shape: (H, W) - already correct
#         prob_map = model_output
#     else:
#         raise ValueError(f"Unexpected model output shape: {model_output.shape}")

#     print(f"  [Postprocess] Probability map shape: {prob_map.shape}")

#     # STEP 2: Apply binary threshold to create binary mask
#     # CRITICAL: OpenCV uses TWO-STAGE thresholding (this is the key!)
#     # Stage 1: Low threshold (0.3) to find all text regions
#     # Stage 2: High threshold (0.7) to get tight text-only pixels
#     #
#     # We create TWO binary maps:
#     # - Low threshold map: For finding contours (includes some background)
#     # - High threshold map: For final box extraction (text only)

#     # CRITICAL FIX: Use the binary_threshold map directly (0.3)
#     # OpenCV uses the low threshold for finding contours to get full text extent
#     # The high threshold approach was making boxes too small vertically
#     binary_map = (prob_map > binary_threshold).astype(np.uint8) * 255

#     print(f"  [Postprocess] Binary threshold: {binary_threshold}")
#     print(f"  [Postprocess] Binary map: {binary_map.shape}, "
#           f"text pixels: {np.sum(binary_map > 0)}")

#     # STEP 2.5: Apply morphological CLOSE to connect text and remove noise
#     # CLOSING = dilation followed by erosion
#     # - Connects nearby characters into words
#     # - Fills small holes in text
#     # - Returns to approximately original size
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # This is our final binary map for contour detection
#     print(f"  [Postprocess] After morphological closing: text pixels: {np.sum(binary_map > 0)}")

#     # STEP 3: Find contours in the binary map
#     # Use RETR_EXTERNAL to get only outer-most contours
#     # This prevents detecting nested contours (like text inside background boxes)
#     # cv2.CHAIN_APPROX_SIMPLE: compress horizontal, vertical, and diagonal segments
#     contours, hierarchy = cv2.findContours(
#         binary_map,
#         cv2.RETR_EXTERNAL,  # Get only external contours (no nested)
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     print(f"  [Postprocess] Found {len(contours)} contours")


#     # Sort contours by area (smallest first)
#     # This prioritizes text regions over background boxes
#     contours = sorted(contours, key=cv2.contourArea)

#     # Lists to store valid boxes and their confidences
#     boxes = []
#     confidences = []
#     processed_regions = []  # Track regions we've already detected

#     # Calculate scaling factors to map from model input size to original image size
#     scale_x = original_size[1] / input_size[0]  # width scaling
#     scale_y = original_size[0] / input_size[1]  # height scaling

#     # STEP 4: Process each contour
#     for contour in contours:
#         # Need at least 4 points for a valid polygon
#         if len(contour) < 4:
#             continue

#         # Calculate polygon area to filter out tiny regions
#         area = cv2.contourArea(contour)
#         if area < 5:  # Lowered from 10 - allow smaller text like single letters
#             continue

#         # STEP 5: Calculate confidence score FIRST (before expanding)
#         # Method: compute mean probability within the original contour region
#         # This ensures we don't go out of bounds when the polygon expands

#         # Create mask for the ORIGINAL contour
#         mask = np.zeros_like(prob_map, dtype=np.uint8)
#         cv2.fillPoly(mask, [contour.astype(np.int32)], 255)

#         # Calculate mean probability in the contour region
#         polygon_region = prob_map[mask > 0]
#         if len(polygon_region) == 0:
#             confidence = 0.0
#             max_prob = 0.0
#         else:
#             confidence = float(np.mean(polygon_region))
#             max_prob = float(np.max(polygon_region))

#         # STEP 6: Filter by polygon threshold
#         if confidence < polygon_threshold:
#             continue  # Discard low-confidence detections

#         # CRITICAL FILTER: Distinguish text from background boxes
#         # Strategy: Use MULTI-CRITERIA filtering for robust detection
#         # - Real text: Most pixels have high probability (tight distribution)
#         # - Background box: Mix of high (text) and medium (background) pixels

#         # Count pixels above different thresholds
#         high_conf_pixels = np.sum(polygon_region > 0.65)
#         total_pixels = len(polygon_region)

#         if total_pixels == 0:
#             continue

#         high_conf_ratio = high_conf_pixels / total_pixels
#         confidence_max = float(np.max(polygon_region))
#         confidence_std = float(np.std(polygon_region))

#         # ENHANCED FILTERING: Multi-criteria confidence analysis
#         # Accept if ANY of these conditions are met:
#         # 1. High ratio of confident pixels (original criterion)
#         # 2. High peak confidence with low variance (consistent detection)
#         # 3. Moderate average confidence with very high peak (clear text core)
#         # 4. Well above polygon threshold (confident detection)
#         accept_detection = (
#             high_conf_ratio >= 0.4 or  # Original criterion
#             (confidence_max > 0.80 and confidence_std < 0.20) or  # Consistent high conf
#             (confidence > 0.40 and confidence_max > 0.85) or  # Strong peak
#             (confidence > polygon_threshold * 1.2)  # Well above threshold
#         )

#         if not accept_detection:
#             continue  # Reject regions that don't meet any criteria

#         # CRITICAL: Apply pyclipper polygon expansion (unclip)
#         # This is ESSENTIAL to the DB algorithm and what OpenCV uses internally
#         # The model outputs SHRUNK text regions (conservative), we expand them back
#         # OPTIMIZED: Using unclip_ratio=4.5 for better expansion (especially for equations)
#         try:
#             expanded_contour = unclip_polygon(contour, unclip_ratio=4.1)
#         except:
#             # If unclipping fails, use original contour
#             expanded_contour = contour

#         # FIX 2: Get AXIS-ALIGNED bounding rectangle (not rotated)
#         # Using cv2.boundingRect instead of cv2.minAreaRect ensures perfectly rectangular boxes
#         x, y, w, h = cv2.boundingRect(expanded_contour)

#         # FIX 1: Add breathing space (delta padding) to avoid tight boundaries
#         # This gives the text some room and prevents clipping at edges
#         delta_w = int(w * delta_width_ratio)  # Configurable width padding
#         delta_h = int(h * delta_height_ratio)  # Configurable height padding

#         # IMPORTANT: Ensure minimum pixel padding for height (since text is usually short)
#         # Without this, small text boxes won't get enough vertical padding
#         delta_h = max(delta_h, min_height_padding_pixels)

#         # Apply padding
#         x = x - delta_w
#         y = y - delta_h
#         w = w + 2 * delta_w
#         h = h + 2 * delta_h

#         # Clip to stay within model input bounds
#         x = max(0, x)
#         y = max(0, y)
#         x_max = min(x + w, input_size[0])
#         y_max = min(y + h, input_size[1])
#         w = x_max - x
#         h = y_max - y

#         # Convert (x, y, w, h) to 4-corner polygon format for consistency
#         # Order: top-left, top-right, bottom-right, bottom-left
#         polygon = np.array([
#             [x, y],              # top-left
#             [x + w, y],          # top-right
#             [x + w, y + h],      # bottom-right
#             [x, y + h]           # bottom-left
#         ], dtype=np.float32)

#         # STEP 7: Scale polygon coordinates back to original image size
#         # Coordinates are currently in model input space (e.g., 320×320)
#         # We need to scale them to original image space
#         polygon_scaled = polygon.astype(np.float32)
#         polygon_scaled[:, 0] *= scale_x  # Scale x coordinates
#         polygon_scaled[:, 1] *= scale_y  # Scale y coordinates

#         # Convert back to integers for final output
#         polygon_scaled = polygon_scaled.astype(np.int32)

#         # Add to results
#         boxes.append(polygon_scaled)
#         confidences.append(confidence)

#         # Mark this region as processed (store unscaled polygon)
#         processed_regions.append(polygon)

#     print(f"  [Postprocess] Filtered to {len(boxes)} valid detections "
#           f"(above threshold {polygon_threshold})")

#     return boxes, confidences


# # ============================================
# # STEP 5: Complete Pipeline (Full Replacement)
# # ============================================

# def detect_text_manual(
#     image: np.ndarray,
#     onnx_session: ort.InferenceSession,
#     config: Dict[str, Any]
# ) -> Tuple[List[np.ndarray], List[float]]:
#     """
#     Complete replacement for textDetectorDB.detect(image).

#     This function combines all steps to provide the same interface as OpenCV's
#     TextDetectionModel_DB.detect() method.

#     Pipeline:
#     1. Preprocess image (resize, normalize, create blob)
#     2. Run inference through ONNX model
#     3. Post-process outputs (threshold, find contours, filter, scale)
#     4. Return boxes and confidences in OpenCV-compatible format

#     Args:
#         image: Input image in BGR format, shape (H, W, 3)
#         onnx_session: Loaded ONNX Runtime inference session
#         config: Configuration dictionary containing:
#             - 'binary_threshold': Threshold for binarization (e.g., 0.3)
#             - 'polygon_threshold': Minimum confidence for detections (e.g., 0.5)
#             - 'input_size': Model input size as (width, height), e.g., (736, 736)
#             - 'mean': Mean values (B, G, R) or (R, G, B) after swap
#             - 'std': Standard deviation for normalization (per-channel)
#             - 'swap_rb': Boolean, whether to swap R and B channels
#             - 'delta_width_ratio': (Optional) Width padding ratio (default: 0.05)
#             - 'delta_height_ratio': (Optional) Height padding ratio (default: 0.10)
#             - 'min_height_padding_pixels': (Optional) Minimum height padding in pixels (default: 5)

#     Returns:
#         boxes: List of detected text boxes as numpy arrays of shape (N, 2)
#                Each box is a polygon: [[x1, y1], [x2, y2], ..., [xN, yN]]
#         confidences: List of confidence scores (0.0 to 1.0) for each box

#     Example:
#         >>> session = load_db_model_manual("DB_TD500_resnet50.onnx")
#         >>> config = {
#         ...     'binary_threshold': 0.3,
#         ...     'polygon_threshold': 0.5,
#         ...     'input_size': (736, 736),
#         ...     'mean': (122.67891434, 116.66876762, 104.00698793),
#         ...     'std': (58.395, 57.12, 57.375),
#         ...     'swap_rb': True
#         ... }
#         >>> image = cv2.imread("test.jpg")
#         >>> boxes, confidences = detect_text_manual(image, session, config)
#         >>> print(f"Detected {len(boxes)} text regions")
#     """
#     print(f"\n{'='*60}")
#     print(f"MANUAL DB TEXT DETECTION PIPELINE")
#     print(f"{'='*60}")

#     # Extract configuration parameters
#     binary_threshold = config['binary_threshold']
#     polygon_threshold = config['polygon_threshold']
#     input_size = config['input_size']
#     mean = config['mean']
#     std = config['std']
#     swap_rb = config['swap_rb']

#     # Extract delta padding parameters (with defaults if not specified)
#     delta_width_ratio = config.get('delta_width_ratio', 0.05)   # Default 5%
#     delta_height_ratio = config.get('delta_height_ratio', 0.10) # Default 10%
#     min_height_padding_pixels = config.get('min_height_padding_pixels', 5)  # Default 5 pixels

#     # STEP 1: Preprocess image
#     print(f"\n[1/3] PREPROCESSING")
#     input_blob, original_size = preprocess_image_manual(
#         image, input_size, mean, std, swap_rb
#     )

#     # STEP 2: Run inference
#     print(f"\n[2/3] INFERENCE")
#     model_output = run_inference_manual(onnx_session, input_blob)

#     # STEP 3: Post-process to get boxes and confidences
#     print(f"\n[3/3] POST-PROCESSING")
#     boxes, confidences = postprocess_db_output_manual(
#         model_output,
#         binary_threshold,
#         polygon_threshold,
#         original_size,
#         input_size,
#         delta_width_ratio,
#         delta_height_ratio,
#         min_height_padding_pixels
#     )

#     print(f"\n{'='*60}")
#     print(f"DETECTION COMPLETE: {len(boxes)} text regions found")
#     print(f"{'='*60}\n")

#     return boxes, confidences


# # ============================================
# # HELPER FUNCTION: Visualize Results
# # ============================================

# def visualize_detections(
#     image: np.ndarray,
#     boxes: List[np.ndarray],
#     confidences: List[float],
#     output_path: str = None
# ) -> np.ndarray:
#     """
#     Draw detected text boxes on image for visualization.

#     Args:
#         image: Input image (BGR format)
#         boxes: List of polygon boxes from detect_text_manual()
#         confidences: List of confidence scores
#         output_path: If provided, save visualization to this path

#     Returns:
#         vis_image: Image with drawn boxes
#     """
#     vis_image = image.copy()

#     for i, (box, conf) in enumerate(zip(boxes, confidences)):
#         # Draw polygon in BLUE (manual detection color)
#         cv2.polylines(vis_image, [box], isClosed=True,
#                      color=(255, 0, 0), thickness=2)

#         # Draw confidence score in BLUE
#         x, y = box[0]
#         text = f"{conf:.2f}"
#         cv2.putText(vis_image, text, (int(x), int(y) - 5),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     if output_path:
#         cv2.imwrite(output_path, vis_image)
#         print(f"Visualization saved to: {output_path}")

#     return vis_image


# # ============================================
# # USAGE EXAMPLE
# # ============================================

# if __name__ == "__main__":
#     """
#     Example showing how to replace OpenCV's TextDetectionModel_DB
#     with the manual implementation.
#     """

#     print("\n" + "="*70)
#     print("DB TEXT DETECTION - MANUAL IMPLEMENTATION")
#     print("Replacing cv2.dnn_TextDetectionModel_DB with ONNX Runtime")
#     print("="*70 + "\n")

#     # ========================================
#     # CONFIGURATION
#     # ========================================

#     # Model path (replace with your actual path)
#     model_path = "model/DB_TD500_resnet50.onnx"

#     # Detection configuration (replaces setBinaryThreshold, setPolygonThreshold, setInputParams)
#     config = {
#         # OPTIMIZED: More sensitive thresholds to catch small text like "C"
#         'binary_threshold': 0.2,     # Lowered from 0.3 - detects fainter text
#         'polygon_threshold': 0.35,   # Lowered from 0.5 - less strict filtering
#         'input_size': (736, 736),    # High resolution for better accuracy
#         'mean': (122.67891434, 116.66876762, 104.00698793),
#         'std': (58.395, 57.12, 57.375),  # Per-channel std normalization
#         'swap_rb': True,

#         # IMPROVED: Better padding for equations and mathematical text
#         'delta_width_ratio': 0.10,   # 10% width padding - generous horizontal space
#         'delta_height_ratio': 0.25,  # 25% height padding - CRITICAL for fractions!
#         'min_height_padding_pixels': 12  # 15 pixels minimum - ensures full coverage
#     }

#     print("Configuration:")
#     for key, value in config.items():
#         print(f"  {key}: {value}")
#     print()

#     # ========================================
#     # OLD WAY (OpenCV's black-box implementation)
#     # ========================================
#     """
#     # This is what we're replacing:
#     textDetector = cv2.dnn_TextDetectionModel_DB(model_path)
#     textDetector.setBinaryThreshold(0.3)
#     textDetector.setPolygonThreshold(0.5)
#     textDetector.setInputParams(std=(58.395, 57.12, 57.375), (736, 736),
#                                 (122.67891434, 116.66876762, 104.00698793),
#                                 True)
#     boxes, confidences = textDetector.detect(image)
#     """

#     # ========================================
#     # NEW WAY (Manual implementation with full control)
#     # ========================================

#     # STEP 1: Load model (do this once at startup)
#     print("STEP 1: Loading ONNX model...")
#     try:
#         session = load_db_model_manual(model_path)
#     except Exception as e:
#         print(f"\nError: {e}")
#         print("\nNote: Please provide the correct path to DB_TD500_resnet50.onnx")
#         print("You can download it from: https://github.com/opencv/opencv_zoo")
#         exit(1)

#     # STEP 2: Load test image
#     print("\nSTEP 2: Loading test image...")
#     image_path = r"model\frame_03324.jpg"  # Replace with your test image

#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             raise FileNotFoundError(f"Could not load image: {image_path}")
#         print(f"[OK] Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
#     except Exception as e:
#         print(f"\nError: {e}")
#         print("\nNote: Please provide a valid test image path")
#         print("Creating a dummy image for demonstration...")
#         image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

#     # STEP 3: Detect text (replaces textDetector.detect(image))
#     print("\nSTEP 3: Detecting text...")
#     boxes, confidences = detect_text_manual(image, session, config)

#     # STEP 4: Display results
#     print("\nRESULTS:")
#     print(f"  Total detections: {len(boxes)}")

#     for i, (box, conf) in enumerate(zip(boxes, confidences)):
#         print(f"  Box {i+1}: {len(box)} vertices, confidence: {conf:.3f}")
#         print(f"    Coordinates: {box[:2]}...")  # Show first 2 vertices

#     # STEP 5: Visualize (optional)
#     if len(boxes) > 0:
#         print("\nSTEP 4: Creating visualization...")
#         vis_image = visualize_detections(image, boxes, confidences,
#                                         output_path="detection_result.jpg")
#         print("[OK] Detection complete!")

#     print("\n" + "="*70)
#     print("COMPARISON: OpenCV vs Manual Implementation")
#     print("="*70)
#     print("""
#     OLD (OpenCV):                    NEW (Manual):
#     ─────────────                    ─────────────

#     textDetector = cv2.dnn_          session = load_db_model_manual(
#         TextDetectionModel_DB(           model_path)
#         model_path)
#                                      config = {
#     textDetector.setBinary               'binary_threshold': 0.3,
#         Threshold(0.3)                   'polygon_threshold': 0.5,
#     textDetector.setPolygon              'input_size': (736, 736),
#         Threshold(0.5)                   'mean': (...),
#     textDetector.setInputParams(         'std': (58.395, 57.12, 57.375),
#         std, (736, 736),                 'swap_rb': True
#         mean, True)                  }

#     boxes, confidences =             boxes, confidences =
#         textDetector.detect(image)       detect_text_manual(
#                                              image, session, config)

#     [OK] Same output format
#     [OK] Same results (boxes + confidences)
#     [OK] Full control over pipeline
#     [OK] Easy to debug and modify
#     """)
#     print("="*70 + "\n")
