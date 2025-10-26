"""
Hate Speech Detection API
A production-ready Flask API for detecting hate speech using a pre-trained ML model.
"""

import os
import pickle
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import traceback
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU-only mode for PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Import torch after setting CPU-only environment
import torch
torch.set_num_threads(1)  # Limit CPU threads for memory efficiency

# CRITICAL: Monkey-patch torch.load to ALWAYS use CPU
_original_torch_load = torch.load
def _cpu_only_load(*args, **kwargs):
    """Force all torch.load calls to use CPU"""
    kwargs['map_location'] = 'cpu'
    return _original_torch_load(*args, **kwargs)
torch.load = _cpu_only_load

logger.info("✓ Torch configured for CPU-only mode")

# Initialize Flask app
app = Flask(__name__)

# CORS configuration for production
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Update with your frontend domain in production
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Production configurations
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Global variable to store the model
model = None
vectorizer = None

def load_model():
    """Load the hate speech detection model - CPU ONLY"""
    global model, vectorizer
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'hate_speech_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from {model_path} (CPU-only mode)")
        
        # Monkey-patch torch.load to always use CPU
        original_torch_load = torch.load
        def cpu_torch_load(*args, **kwargs):
            kwargs['map_location'] = 'cpu'
            return original_torch_load(*args, **kwargs)
        
        torch.load = cpu_torch_load
        
        try:
            # Load with patched torch.load
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
        # Handle different model storage formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            vectorizer = model_data.get('vectorizer')
            logger.info("Model format: dictionary")
        elif isinstance(model_data, tuple):
            model, vectorizer = model_data
            logger.info("Model format: tuple")
        else:
            model = model_data
            vectorizer = None
            logger.info("Model format: single object")
        
        # Force model to CPU and eval mode
        if hasattr(model, 'to'):
            model.to('cpu')
            model.eval()
            logger.info("Model set to CPU and eval mode")
        
        logger.info("✓ Model loaded successfully on CPU")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model on startup
if not load_model():
    logger.warning("Failed to load model on startup. Will retry on first request.")

# Error handlers
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handle HTTP exceptions"""
    response = {
        "error": e.name,
        "message": e.description,
        "status": e.code
    }
    return jsonify(response), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle uncaught exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    
    response = {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later.",
        "status": 500
    }
    return jsonify(response), 500

# API Routes
@app.route('/', methods=['GET'])
def home():
    """Home endpoint - API information"""
    return jsonify({
        "service": "Hate Speech Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/api/predict",
            "batch_predict": "/api/batch-predict"
        },
        "documentation": "/api/docs"
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    model_status = "loaded" if model is not None else "not_loaded"
    vectorizer_status = "loaded" if vectorizer is not None else "not_loaded"
    
    health_status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_status": model_status,
        "vectorizer_status": vectorizer_status,
        "timestamp": os.environ.get('RENDER_GIT_COMMIT', 'unknown')
    }
    
    status_code = 200 if model is not None else 503
    return jsonify(health_status), status_code

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    docs = {
        "api_name": "Hate Speech Detection API",
        "version": "1.0.0",
        "description": "API for detecting hate speech in text using machine learning",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "API information and available endpoints"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint for monitoring service status"
            },
            {
                "path": "/api/predict",
                "method": "POST",
                "description": "Predict hate speech for a single text input",
                "request_body": {
                    "text": "string (required) - The text to analyze"
                },
                "response": {
                    "text": "The analyzed text",
                    "prediction": "The prediction result (0 or 1)",
                    "label": "Human-readable label (normal/hate_speech)",
                    "confidence": "Confidence score (if available)",
                    "status": "success"
                }
            },
            {
                "path": "/api/batch-predict",
                "method": "POST",
                "description": "Predict hate speech for multiple texts",
                "request_body": {
                    "texts": ["array of strings (required) - List of texts to analyze"]
                },
                "response": {
                    "predictions": "Array of prediction results",
                    "count": "Number of texts analyzed",
                    "status": "success"
                }
            }
        ],
        "error_responses": {
            "400": "Bad Request - Invalid input",
            "500": "Internal Server Error",
            "503": "Service Unavailable - Model not loaded"
        }
    }
    return jsonify(docs), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict hate speech for a single text input"""
    global model, vectorizer
    
    # Retry loading model if not loaded
    if model is None:
        logger.warning("Model not loaded, attempting to reload")
        if not load_model():
            return jsonify({
                "error": "Model not available",
                "message": "The model could not be loaded. Please try again later.",
                "status": 503
            }), 503
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Invalid request",
                "message": "Content-Type must be application/json",
                "status": 400
            }), 400
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing required field",
                "message": "Request must include 'text' field",
                "status": 400
            }), 400
        
        text = data['text']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                "error": "Invalid input",
                "message": "Text must be a non-empty string",
                "status": 400
            }), 400
        
        # Preprocess and predict (CPU-only inference)
        with torch.no_grad():  # Disable gradient computation for inference
            if vectorizer is not None:
                text_vectorized = vectorizer.transform([text])
                prediction = model.predict(text_vectorized)[0]
                
                # Get probability if available
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(text_vectorized)[0]
                    confidence = float(max(proba))
            else:
                # If no vectorizer, assume model handles raw text
                prediction = model.predict([text])[0]
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([text])[0]
                    confidence = float(max(proba))
        
        # Prepare response
        result = {
            "text": text,
            "prediction": int(prediction),
            "label": "hate_speech" if prediction == 1 else "normal",
            "status": "success"
        }
        
        if confidence is not None:
            result["confidence"] = confidence
        
        logger.info(f"Prediction made: {result['label']} (confidence: {confidence})")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "status": 500
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict hate speech for multiple texts"""
    global model, vectorizer
    
    # Retry loading model if not loaded
    if model is None:
        logger.warning("Model not loaded, attempting to reload")
        if not load_model():
            return jsonify({
                "error": "Model not available",
                "message": "The model could not be loaded. Please try again later.",
                "status": 503
            }), 503
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Invalid request",
                "message": "Content-Type must be application/json",
                "status": 400
            }), 400
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing required field",
                "message": "Request must include 'texts' field (array of strings)",
                "status": 400
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "error": "Invalid input",
                "message": "Texts must be a non-empty array",
                "status": 400
            }), 400
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return jsonify({
                    "error": "Invalid input",
                    "message": f"Text at index {i} must be a non-empty string",
                    "status": 400
                }), 400
        
        # Limit batch size
        if len(texts) > 100:
            return jsonify({
                "error": "Batch size exceeded",
                "message": "Maximum 100 texts per batch request",
                "status": 400
            }), 400
        
        # Preprocess and predict (CPU-only inference)
        results = []
        
        with torch.no_grad():  # Disable gradient computation for inference
            if vectorizer is not None:
                texts_vectorized = vectorizer.transform(texts)
                predictions = model.predict(texts_vectorized)
                
                # Get probabilities if available
                confidences = None
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(texts_vectorized)
                    confidences = [float(max(proba)) for proba in probas]
            else:
                predictions = model.predict(texts)
                confidences = None
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(texts)
                    confidences = [float(max(proba)) for proba in probas]
        
        # Prepare results
        for i, (text, prediction) in enumerate(zip(texts, predictions)):
            result = {
                "text": text,
                "prediction": int(prediction),
                "label": "hate_speech" if prediction == 1 else "normal"
            }
            if confidences is not None:
                result["confidence"] = confidences[i]
            results.append(result)
        
        response = {
            "predictions": results,
            "count": len(results),
            "status": "success"
        }
        
        logger.info(f"Batch prediction made for {len(texts)} texts")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Batch prediction failed",
            "message": str(e),
            "status": 500
        }), 500

# Production server configuration
if __name__ == '__main__':
    # This will only run in development
    # In production, Gunicorn will be used instead
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
