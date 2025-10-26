# Hate Speech Detection API

A production-ready Flask REST API for detecting hate speech in text using machine learning. This API is designed for deployment on Render and provides real-time hate speech classification.

## 🚀 Features

- **Production-Ready**: Configured with Gunicorn, proper logging, and error handling
- **CORS Enabled**: Ready for cross-origin requests from frontend applications
- **Health Monitoring**: Built-in health check endpoint for service monitoring
- **Batch Processing**: Support for single and batch text predictions
- **Error Handling**: Comprehensive error handling with meaningful responses
- **API Documentation**: Built-in API documentation endpoint

## 📋 API Endpoints

### Base URL
```
Production: https://your-app-name.onrender.com
Local: http://localhost:5000
```

### 1. Home / Service Info
**Endpoint**: `GET /`

**Description**: Get API information and available endpoints

**Response**:
```json
{
  "service": "Hate Speech Detection API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "predict": "/api/predict",
    "batch_predict": "/api/batch-predict"
  },
  "documentation": "/api/docs"
}
```

---

### 2. Health Check
**Endpoint**: `GET /health`

**Description**: Check service health and model status

**Response**:
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "vectorizer_status": "loaded",
  "timestamp": "abc123"
}
```

**Status Codes**:
- `200`: Service is healthy
- `503`: Service is unhealthy (model not loaded)

---

### 3. API Documentation
**Endpoint**: `GET /api/docs`

**Description**: Get comprehensive API documentation

**Response**: JSON object with detailed endpoint information

---

### 4. Single Text Prediction
**Endpoint**: `POST /api/predict`

**Description**: Predict hate speech for a single text input

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "text": "Your text to analyze here"
}
```

**Response** (Success - 200):
```json
{
  "text": "Your text to analyze here",
  "prediction": 0,
  "label": "normal",
  "confidence": 0.95,
  "status": "success"
}
```

**Response Fields**:
- `text`: The input text that was analyzed
- `prediction`: Numeric prediction (0 = normal, 1 = hate speech)
- `label`: Human-readable label ("normal" or "hate_speech")
- `confidence`: Confidence score (0-1, if available)
- `status`: Request status

**Error Responses**:

400 - Bad Request:
```json
{
  "error": "Invalid input",
  "message": "Text must be a non-empty string",
  "status": 400
}
```

503 - Service Unavailable:
```json
{
  "error": "Model not available",
  "message": "The model could not be loaded. Please try again later.",
  "status": 503
}
```

**cURL Example**:
```bash
curl -X POST https://your-app-name.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text to analyze"}'
```

**Python Example**:
```python
import requests

url = "https://your-app-name.onrender.com/api/predict"
payload = {"text": "This is a sample text to analyze"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result)
```

**JavaScript Example**:
```javascript
const url = "https://your-app-name.onrender.com/api/predict";
const payload = { text: "This is a sample text to analyze" };

fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error("Error:", error));
```

---

### 5. Batch Text Prediction
**Endpoint**: `POST /api/batch-predict`

**Description**: Predict hate speech for multiple texts (max 100 per request)

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "texts": [
    "First text to analyze",
    "Second text to analyze",
    "Third text to analyze"
  ]
}
```

**Response** (Success - 200):
```json
{
  "predictions": [
    {
      "text": "First text to analyze",
      "prediction": 0,
      "label": "normal",
      "confidence": 0.92
    },
    {
      "text": "Second text to analyze",
      "prediction": 1,
      "label": "hate_speech",
      "confidence": 0.88
    },
    {
      "text": "Third text to analyze",
      "prediction": 0,
      "label": "normal",
      "confidence": 0.95
    }
  ],
  "count": 3,
  "status": "success"
}
```

**Error Responses**:

400 - Batch Size Exceeded:
```json
{
  "error": "Batch size exceeded",
  "message": "Maximum 100 texts per batch request",
  "status": 400
}
```

**cURL Example**:
```bash
curl -X POST https://your-app-name.onrender.com/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

**Python Example**:
```python
import requests

url = "https://your-app-name.onrender.com/api/batch-predict"
payload = {
    "texts": [
        "First text to analyze",
        "Second text to analyze",
        "Third text to analyze"
    ]
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(f"Analyzed {result['count']} texts")
for pred in result['predictions']:
    print(f"Text: {pred['text'][:50]}... -> {pred['label']}")
```

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Flask_CC
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Ensure your `hate_speech_model.pkl` file is in the project root

6. Run the application:
```bash
# Development mode
python app.py

# Production mode with Gunicorn
gunicorn app:app --workers 4 --bind 0.0.0.0:5000
```

7. Test the API:
```bash
# Test health check
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Test message\"}"
```

---

## 🚢 Deployment on Render

### Method 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. **Deploy on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and configure everything
   - Click "Apply" to deploy

### Method 2: Manual Setup

1. **Push your code to GitHub** (same as above)

2. **Create a new Web Service on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:

   **Settings**:
   - **Name**: `hate-speech-api` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 4 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Plan**: Free (or paid for production)

3. **Environment Variables** (Optional):
   - No environment variables are required by default
   - Add custom variables if needed (e.g., API keys, custom configurations)

4. **Deploy**:
   - Click "Create Web Service"
   - Render will build and deploy your application
   - Wait for deployment to complete (usually 5-10 minutes)

5. **Access your API**:
   - Your API will be available at: `https://your-app-name.onrender.com`
   - Test with: `https://your-app-name.onrender.com/health`

### Important Notes for Render Deployment

⚠️ **Free Tier Limitations**:
- Free services spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- For production, consider paid plans for 24/7 availability

📦 **Model File**:
- Ensure `hate_speech_model.pkl` is committed to your repository
- Large model files (>100MB) may need Git LFS or external storage
- Check Render's disk space limits for your plan

🔒 **Security**:
- Update CORS origins in `app.py` to restrict to your frontend domain
- Add authentication if needed for production use
- Consider rate limiting for public APIs

---

## 📊 Model Format

The API expects the model pickle file to contain either:

1. **Dictionary format**:
```python
{
    'model': trained_model,
    'vectorizer': text_vectorizer
}
```

2. **Tuple format**:
```python
(trained_model, text_vectorizer)
```

3. **Model only** (if it handles raw text):
```python
trained_model
```

---

## 🔧 Configuration

### Production Settings (Already Configured)

The application includes these production-ready configurations:

- **Workers**: 4 Gunicorn workers with 2 threads each
- **Timeout**: 120 seconds for long-running predictions
- **Logging**: INFO level with structured logging
- **CORS**: Enabled for all origins (customize in `app.py`)
- **Max Request Size**: 16MB
- **Health Checks**: Automatic health monitoring

### Customization

To customize the API, edit `app.py`:

```python
# Update CORS origins for production
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://your-frontend-domain.com"],
        # ... other settings
    }
})

# Adjust max request size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Modify batch size limit
if len(texts) > 100:  # Change 100 to your desired limit
    # ...
```

---

## 📝 Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Endpoint doesn't exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

---

## 🧪 Testing

### Manual Testing

Use these commands to test all endpoints:

```bash
# Set your base URL
BASE_URL="https://your-app-name.onrender.com"

# Test home endpoint
curl $BASE_URL/

# Test health check
curl $BASE_URL/health

# Test API documentation
curl $BASE_URL/api/docs

# Test single prediction
curl -X POST $BASE_URL/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message"}'

# Test batch prediction
curl -X POST $BASE_URL/api/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["First message", "Second message", "Third message"]}'
```

### Integration with Frontend

To integrate with your frontend application:

```javascript
// Create an API client
class HateSpeechAPI {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }

  async predict(text) {
    const response = await fetch(`${this.baseURL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    return await response.json();
  }

  async batchPredict(texts) {
    const response = await fetch(`${this.baseURL}/api/batch-predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    return await response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    return await response.json();
  }
}

// Usage
const api = new HateSpeechAPI('https://your-app-name.onrender.com');

// Single prediction
const result = await api.predict('Your text here');
console.log(result);

// Batch prediction
const results = await api.batchPredict(['Text 1', 'Text 2', 'Text 3']);
console.log(results);

// Health check
const health = await api.healthCheck();
console.log(health);
```

---

## 📚 Tech Stack

- **Framework**: Flask 3.0.0
- **WSGI Server**: Gunicorn 21.2.0
- **ML Libraries**: scikit-learn, NumPy, SciPy
- **CORS**: Flask-CORS 4.0.0
- **Python Version**: 3.11

---

## 🤝 Support

For issues or questions:
1. Check the health endpoint: `/health`
2. Review logs in Render dashboard
3. Verify model file is properly loaded
4. Ensure all dependencies are installed

---

## 📄 License

This project is configured for production deployment on Render.

---

## 🔄 Updates and Maintenance

To update your deployed application:

1. Make changes to your code
2. Commit and push to GitHub:
```bash
git add .
git commit -m "Your update message"
git push
```
3. Render will automatically rebuild and redeploy

To manually trigger a deployment:
- Go to Render Dashboard → Your Service → "Manual Deploy" → "Deploy latest commit"

---

## 🎯 Quick Start Checklist

- [ ] Clone/create project directory
- [ ] Ensure `hate_speech_model.pkl` is in the root
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test locally: `python app.py`
- [ ] Initialize Git repository
- [ ] Push to GitHub
- [ ] Create Web Service on Render
- [ ] Configure build and start commands
- [ ] Deploy and test production URL
- [ ] Update CORS settings with your frontend domain
- [ ] Monitor health endpoint for service status

---

## 🌟 Production Best Practices

✅ **Implemented**:
- Production WSGI server (Gunicorn)
- Proper error handling and logging
- Health check endpoint for monitoring
- CORS configuration
- Request size limits
- Batch processing support
- Comprehensive API documentation

📝 **Recommended**:
- Set up monitoring and alerting
- Implement rate limiting
- Add authentication for sensitive use cases
- Use environment variables for configuration
- Set up CI/CD pipeline
- Implement caching for frequently requested predictions
- Add request validation middleware
- Set up backup strategies for the model file

---

**Your API is now ready for production deployment on Render! 🚀**
