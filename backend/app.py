"""
Flask backend for MediSync FL - Federated Learning Medical Image Analysis
Serves ML model predictions and provides REST API for React frontend
"""

import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf_generator import PredictionReport

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

MODEL_PATH = MODEL_DIR / 'global_model.pth'
LABEL_MAP_PATH = MODEL_DIR / 'label_map.json'
META_PATH = MODEL_DIR / 'model_meta.json'
DATASET_STATS_PATH = ARTIFACTS_DIR / 'dataset_stats.json'
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / 'training_history.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global model cache
model_cache = {'model': None, 'label_map': None, 'idx_to_label': None}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_label_map():
    """Load model label mapping"""
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
        idx_to_label = {int(v): k for k, v in label_map.items()}
        logger.info('Loaded label map')
        return label_map, idx_to_label
    logger.warning('Label map not found')
    return None, None


def load_dataset_stats():
    """Load dataset statistics"""
    if DATASET_STATS_PATH.exists():
        try:
            with open(DATASET_STATS_PATH, 'r') as f:
                stats = json.load(f)
            logger.info('Loaded dataset statistics')
            return stats
        except Exception as e:
            logger.error(f'Error loading dataset stats: {e}')
    return None


def load_model_meta():
    """Load model metadata"""
    if META_PATH.exists():
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            logger.info('Loaded model metadata')
            return meta
        except Exception as e:
            logger.error(f'Error loading model metadata: {e}')
    return None


def load_training_history():
    """Load training history"""
    if TRAINING_HISTORY_PATH.exists():
        try:
            with open(TRAINING_HISTORY_PATH, 'r') as f:
                history = json.load(f)
            logger.info('Loaded training history')
            return history
        except Exception as e:
            logger.error(f'Error loading training history: {e}')
    return None


def load_model():
    """Load model with caching"""
    if model_cache['model'] is not None:
        return model_cache['model'], model_cache['label_map'], model_cache['idx_to_label']

    label_map, idx_to_label = load_label_map()
    if label_map is None:
        logger.error('Cannot load model without label map')
        return None, None, None

    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(label_map))
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            logger.info('Loaded trained model weights')
        else:
            logger.warning('Model weights not found, using untrained model')
        model.to(DEVICE)
        model.eval()
        
        model_cache['model'] = model
        model_cache['label_map'] = label_map
        model_cache['idx_to_label'] = idx_to_label
        
        return model, label_map, idx_to_label
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        return None, None, None


def predict(image: Image.Image, model, idx_to_label):
    """Run inference on image"""
    logger.info('Running inference on uploaded image')
    try:
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        
        results = {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
        top_label = max(results, key=results.get)
        logger.info(f'Prediction: {top_label} ({results[top_label]*100:.2f}%)')
        return top_label, results
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        return None, None


# Routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(DEVICE)}), 200


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model metadata"""
    try:
        meta = load_model_meta()
        if not meta:
            return jsonify({'error': 'Model metadata not found'}), 404
        
        return jsonify({
            'trained_at': meta.get('trained_at'),
            'device': str(DEVICE),
            'num_epochs': meta.get('num_epochs'),
            'best_epoch': meta.get('best_epoch'),
            'metrics': meta.get('metrics', {}),
            'total_samples': meta.get('total_samples')
        }), 200
    except Exception as e:
        logger.error(f'Error fetching model info: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/network-dashboard', methods=['GET'])
def network_dashboard():
    """Get network dashboard data"""
    try:
        dataset_stats = load_dataset_stats()
        meta = load_model_meta()
        
        if not dataset_stats or not meta:
            return jsonify({'error': 'Required data not found'}), 404
        
        total_hospitals = len(dataset_stats)
        total_patients = meta.get('total_samples', 0)
        
        return jsonify({
            'total_hospitals': total_hospitals,
            'total_patients': total_patients,
            'global_accuracy': meta['metrics'].get('test_accuracy', 0) * 100,
            'best_val_accuracy': meta['metrics'].get('best_val_accuracy', 0) * 100,
            'avg_f1': meta['metrics'].get('avg_f1', 0) * 100,
            'avg_precision': meta['metrics'].get('avg_precision', 0) * 100,
            'avg_recall': meta['metrics'].get('avg_recall', 0) * 100,
            'num_epochs': meta.get('num_epochs', 0),
            'best_epoch': meta.get('best_epoch', 0),
            'hospitals': dataset_stats
        }), 200
    except Exception as e:
        logger.error(f'Error fetching dashboard data: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Get analytics data"""
    try:
        history = load_training_history()
        meta = load_model_meta()
        dataset_stats = load_dataset_stats()
        
        if not history or not meta or not dataset_stats:
            return jsonify({'error': 'Required data not found'}), 404
        
        # Hospital contribution
        hospital_names = list(dataset_stats.keys())
        contributions = [info['total_samples'] for info in dataset_stats.values()]
        
        # Aggregate class distribution
        all_classes = {}
        for hospital_info in dataset_stats.values():
            for cls, count in hospital_info.get('class_distribution', {}).items():
                all_classes[cls] = all_classes.get(cls, 0) + count
        
        return jsonify({
            'training_history': history,
            'test_metrics': meta.get('metrics', {}),
            'hospital_contributions': {
                'names': hospital_names,
                'values': contributions
            },
            'class_distribution': all_classes
        }), 200
    except Exception as e:
        logger.error(f'Error fetching analytics: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """Prediction endpoint"""
    try:
        # Check file upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Load model
        model, label_map, idx_to_label = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Process image
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            logger.error(f'Error opening image: {e}')
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Get image properties
        image_np = np.array(image)
        mean_luma = float(image_np.mean())
        std_luma = float(image_np.std())
        height, width = image_np.shape[:2]
        
        # Run prediction
        top_label, results = predict(image, model, idx_to_label)
        
        if results is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Prepare response
        top_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
        meta = load_model_meta()
        
        return jsonify({
            'predicted_class': top_label.upper(),
            'confidence': float(results[top_label]),
            'timestamp': datetime.utcnow().isoformat(),
            'image_properties': {
                'resolution': f'{width}x{height}',
                'brightness': round(mean_luma, 2),
                'contrast': round(std_luma, 2)
            },
            'all_predictions': {label: round(prob, 4) for label, prob in top_sorted},
            'device': str(DEVICE),
            'model_info': {
                'trained_at': meta.get('trained_at') if meta else None,
                'test_accuracy': round(meta['metrics'].get('test_accuracy', 0) * 100, 2) if meta else None
            }
        }), 200
    
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-pdf', methods=['POST'])
def predict_pdf():
    """Get prediction and return as PDF report"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Load model
        model, label_map, idx_to_label = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Process image
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            logger.error(f'Error opening image: {e}')
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Run prediction
        top_label, results = predict(image, model, idx_to_label)
        
        if results is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get model info
        meta = load_model_meta()
        model_accuracy = float(meta['metrics'].get('test_accuracy', 0.78)) if meta else 0.78
        
        # Prepare prediction data for PDF
        prediction_data = {
            'predicted_class': top_label.lower(),
            'confidence': float(results[top_label]),
            'all_predictions': {label: round(prob, 4) for label, prob in results.items()},
            'filename': secure_filename(file.filename),
            'timestamp': datetime.utcnow().isoformat(),
            'model_accuracy': model_accuracy
        }
        
        # Generate PDF
        pdf_generator = PredictionReport()
        pdf_content = pdf_generator.generate(prediction_data)
        
        logger.info(f'Generated PDF report for {top_label}')
        
        # Return PDF file
        return send_file(
            BytesIO(pdf_content),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        logger.error(f'PDF generation error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/training-history', methods=['GET'])
def training_history():
    """Get training history"""
    try:
        history = load_training_history()
        if not history:
            return jsonify({'error': 'Training history not found'}), 404
        return jsonify({'history': history}), 200
    except Exception as e:
        logger.error(f'Error fetching training history: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/hospital/<hospital_name>', methods=['GET'])
def hospital_details(hospital_name):
    """Get specific hospital details"""
    try:
        dataset_stats = load_dataset_stats()
        if not dataset_stats or hospital_name not in dataset_stats:
            return jsonify({'error': 'Hospital not found'}), 404
        
        return jsonify(dataset_stats[hospital_name]), 200
    except Exception as e:
        logger.error(f'Error fetching hospital details: {e}')
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f'Server error: {e}')
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info(f'Starting MediSync FL Backend on {DEVICE}')
    app.run(debug=True, host='0.0.0.0', port=6060)
