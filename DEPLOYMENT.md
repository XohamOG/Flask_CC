# 🚀 DEPLOYMENT GUIDE - Render

## Step-by-Step Deployment Instructions

### Step 1: Push Your Code to GitHub

1. Open PowerShell in your project directory (d:\Flask_CC)

2. Initialize Git repository:
```powershell
git init
```

3. Add all files:
```powershell
git add .
```

4. Commit your files:
```powershell
git commit -m "Initial commit - Hate Speech Detection API"
```

5. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name it (e.g., "hate-speech-api")
   - Don't initialize with README (we already have one)
   - Click "Create repository"

6. Link your local repo to GitHub:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

#### Option A: Using Blueprint (Recommended - Automatic)

1. Go to https://dashboard.render.com/
   - Sign up or log in

2. Click "New +" button → Select "Blueprint"

3. Connect your GitHub account (if not already connected)
   - Click "Connect GitHub"
   - Authorize Render

4. Select your repository:
   - Find "hate-speech-api" (or your repo name)
   - Click "Connect"

5. Render will detect your `render.yaml` file automatically
   - Review the configuration
   - Click "Apply"

6. Wait for deployment (5-10 minutes)
   - Watch the build logs
   - Once complete, your API will be live!

#### Option B: Manual Setup

1. Go to https://dashboard.render.com/

2. Click "New +" → Select "Web Service"

3. Connect your repository:
   - Click "Connect GitHub"
   - Select your repository

4. Configure the service:
   - **Name**: `hate-speech-api` (or your choice)
   - **Region**: Choose closest to you (e.g., Oregon, Ohio, Frankfurt)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 4 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`

5. Choose your plan:
   - **Free**: Good for testing (spins down after 15 min inactivity)
   - **Starter** ($7/mo): Recommended for production (always on)

6. Click "Create Web Service"

7. Wait for deployment (5-10 minutes)

### Step 3: Verify Deployment

1. Once deployed, Render will give you a URL like:
   ```
   https://hate-speech-api-xxxx.onrender.com
   ```

2. Test the health endpoint:
   ```powershell
   curl https://your-app-name.onrender.com/health
   ```

3. Test the prediction endpoint:
   ```powershell
   curl -X POST https://your-app-name.onrender.com/api/predict -H "Content-Type: application/json" -d '{\"text\": \"test message\"}'
   ```

### Step 4: Update Your Frontend

Update your frontend to use the new API URL:
```javascript
const API_URL = "https://your-app-name.onrender.com";
```

## 🔧 Troubleshooting

### Build Failed?
- Check build logs in Render dashboard
- Ensure `requirements.txt` is correct
- Verify `hate_speech_model.pkl` was pushed to GitHub

### Model not loading?
- Check if `hate_speech_model.pkl` exists in repository
- If file is >100MB, you may need Git LFS
- Check logs for model loading errors

### Service Unavailable (503)?
- Model might be too large or taking time to load
- Check health endpoint after a few minutes
- Review logs in Render dashboard

### Free Tier Spinning Down?
- First request after 15 min may take 30-60 seconds
- Upgrade to Starter plan for always-on service

## 📊 Monitoring Your API

- **Dashboard**: https://dashboard.render.com/
- **Logs**: Click your service → "Logs" tab
- **Metrics**: Click your service → "Metrics" tab
- **Health**: Visit `/health` endpoint anytime

## 🔄 Making Updates

To update your deployed API:

1. Make changes to your code locally

2. Commit and push:
```powershell
git add .
git commit -m "Description of changes"
git push
```

3. Render will automatically redeploy!

## 🎯 Your API is Ready!

Once deployed, you can access:
- **API Base**: `https://your-app-name.onrender.com/`
- **Health Check**: `https://your-app-name.onrender.com/health`
- **Documentation**: `https://your-app-name.onrender.com/api/docs`
- **Predict**: `POST https://your-app-name.onrender.com/api/predict`
