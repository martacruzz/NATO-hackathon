import numpy as np
import tensorflow as tf
import cv2
import os
import argparse

# Configuration
CLASSES = [
    "Background, including WiFi and Bluetooth",
    "DJI Phantom 3",
    "DJI Phantom 4 Rro",
    "DJI MATRICE 200",
    "DJI MATRICE 100",
    "DJI Air 2S",
    "DJI Mini 3 Pro",
    "DJI Inspire 2",
    "DJI Mavic Pro",
    "DJI Mini 2",
    "DJI Mavic 3",
    "DJI MATRICE 300",
    "DJI Phantom 4 Pro RTK",
    "DJI MATRICE 30T",
    "DJI AVATA",
    "DJI DIY",
    "DJI MATRICE 600 Pro",
    "VBar",
    "FrSky X20",
    "Futaba T16IZ",
    "Taranis Plus",
    "RadioLink AT9S",
    "Futaba T14SG",
    "Skydroid"
]

def load_model_and_embeddings():
    """Load model and embeddings from disk with proper building"""
    model = tf.keras.models.load_model('final_drone_classifier.keras')
    
    # Force build the model with dummy input
    dummy_input = tf.zeros((1, 256, 256, 1))
    model(dummy_input)
    
    # Load embeddings
    class_centers = np.load('class_centers.npy')
    class_thresholds = np.load('class_thresholds.npy')
    
    # Create embedding model using first layer's input
    embedding_layer = model.get_layer('embedding_layer')
    embedding_model = tf.keras.Model(
        inputs=model.layers[0].input,  # Use first layer's input tensor
        outputs=embedding_layer.output
    )
    
    return model, class_centers, class_thresholds, embedding_model

def preprocess_spectrogram(spectrogram_path):
    """Preprocess a spectrogram for prediction"""
    # Load the .npy file
    spec = np.load(spectrogram_path)
    
    # Resize to 256x256
    spec = cv2.resize(spec.astype(np.float32), (256, 256))
    
    # Add channel dimension
    spec = np.expand_dims(spec, axis=-1)
    
    # Normalize to [0, 1]
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    
    return spec

def predict_drone(model, class_centers, class_thresholds, embedding_model, spectrogram_path, confidence_threshold=0.5):
    """
    Predict drone class for a spectrogram
    
    Args:
        model: Trained TensorFlow model
        class_centers: Precomputed class centers
        class_thresholds: Thresholds for each class
        embedding_model: Model that outputs embedding layer
        spectrogram_path: Path to .npy spectrogram file
        confidence_threshold: Minimum confidence to accept a known class
        
    Returns:
        (class_name, confidence_score)
    """
    # Preprocess
    spec = preprocess_spectrogram(spectrogram_path)
    
    # Get model prediction
    prediction = model.predict(np.expand_dims(spec, axis=0))[0]
    class_idx = np.argmax(prediction)
    model_confidence = prediction[class_idx]
    
    # Get embedding using the embedding model
    embedding = embedding_model.predict(np.expand_dims(spec, axis=0))
    
    # Calculate distance to class center
    dist = np.linalg.norm(embedding - class_centers[class_idx])
    
    # Check if unknown
    if dist > class_thresholds[class_idx]:
        return "Unknown", 0.0
    
    # Calculate confidence score from distance
    distance_confidence = 1.0 - (dist / class_thresholds[class_idx])
    
    # Use the higher of the two confidence scores
    final_confidence = max(model_confidence, distance_confidence)
    
    # Apply confidence threshold
    if final_confidence < confidence_threshold:
        return "Unknown", 0.0
        
    return CLASSES[class_idx], final_confidence

def evaluate_dataset(data_dir, confidence_threshold=0.5):
    """
    Evaluate model performance on a dataset
    
    Args:
        data_dir: Directory with subdirectories named by class index
        confidence_threshold: Minimum confidence to accept a known class
        
    Returns:
        Dictionary of evaluation metrics
    """
    true_labels = []
    pred_labels = []
    
    # Load model and embeddings once
    model, class_centers, class_thresholds, embedding_model = load_model_and_embeddings()
    
    # Iterate through each class directory
    for class_idx in range(len(CLASSES)):
        class_dir = os.path.join(data_dir, str(class_idx))
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if filename.endswith('.npy'):
                spectrogram_path = os.path.join(class_dir, filename)
                predicted_class, _ = predict_drone(
                    model, class_centers, class_thresholds, 
                    embedding_model, spectrogram_path, confidence_threshold
                )
                
                # Convert predicted class to index
                if predicted_class == "Unknown":
                    pred_idx = -1
                else:
                    pred_idx = CLASSES.index(predicted_class)
                
                true_labels.append(class_idx)
                pred_labels.append(pred_idx)
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Calculate accuracy
    accuracy = np.mean(true_labels == pred_labels)
    
    # Confusion matrix (including unknown class)
    num_classes = len(CLASSES) + 1  # +1 for unknown
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(true_labels)):
        t = true_labels[i]
        p = pred_labels[i]
        confusion_matrix[t, p] += 1
    
    # Precision and Recall for each class
    precision = []
    recall = []
    for i in range(num_classes):
        true_pos = confusion_matrix[i, i]
        pred_pos = np.sum(confusion_matrix[:, i])
        actual_pos = np.sum(confusion_matrix[i, :])
        
        p = true_pos / pred_pos if pred_pos > 0 else 0.0
        r = true_pos / actual_pos if actual_pos > 0 else 0.0
        
        precision.append(p)
        recall.append(r)
    
    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'confusion_matrix': confusion_matrix,
        'class_precision': precision,
        'class_recall': recall
    }

def print_evaluation_results(results):
    """Print evaluation results in a readable format"""
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    
    print("\n🔍 Per-class Metrics:")
    for i, class_name in enumerate(CLASSES):
        print(f"\n{class_name}:")
        print(f"  Precision: {results['class_precision'][i]:.4f}")
        print(f"  Recall: {results['class_recall'][i]:.4f}")
    
    # Unknown class metrics
    unknown_idx = len(CLASSES)
    print(f"\nUnknown Class:")
    print(f"  Precision: {results['class_precision'][unknown_idx]:.4f}")
    print(f"  Recall: {results['class_recall'][unknown_idx]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Drone Detection Model Tester')
    parser.add_argument('--single', type=str, help='Path to single spectrogram for testing')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory (with class subdirectories)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for known classes')
    args = parser.parse_args()
    
    # For single file prediction, load model and embeddings
    if args.single:
        model, class_centers, class_thresholds, embedding_model = load_model_and_embeddings()
        class_name, confidence = predict_drone(
            model, class_centers, class_thresholds, 
            embedding_model, args.single, args.confidence
        )
        print(f"\nPrediction for {args.single}:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {confidence:.4f}")
    
    # For dataset evaluation, load model and embeddings once
    if args.dataset:
        results = evaluate_dataset(args.dataset, args.confidence)
        print_evaluation_results(results)
    
    # If no arguments provided, show help
    if not args.single and not args.dataset:
        parser.print_help()

if __name__ == "__main__":
    main()