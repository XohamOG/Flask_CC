# 🔧 Git LFS Setup Guide

## What is Git LFS?
Git Large File Storage (LFS) is used to handle large files (>100MB) in Git repositories. It stores large files on a separate server and keeps only references in your Git repo.

## Step-by-Step Setup

### 1. Install Git LFS

**Windows (using Chocolatey):**
```powershell
choco install git-lfs
```

**Windows (using Scoop):**
```powershell
scoop install git-lfs
```

**Windows (Manual Download):**
1. Download from: https://git-lfs.github.com/
2. Run the installer
3. Restart PowerShell

**Verify Installation:**
```powershell
git lfs version
```
You should see something like: `git-lfs/3.x.x`

### 2. Initialize Git LFS

Run in your project directory (d:\Flask_CC):

```powershell
# Initialize Git LFS for your user account (only needed once)
git lfs install

# This will output: "Git LFS initialized."
```

### 3. Track Your Model File

The `.gitattributes` file is already created and configured to track .pkl files.

Verify it's set up:
```powershell
cat .gitattributes
```

You should see:
```
*.pkl filter=lfs diff=lfs merge=lfs -text
```

### 4. Add and Commit Your Files

Now add your files to Git:

```powershell
# Remove the model from regular Git cache if it was already added
git rm --cached hate_speech_model.pkl

# Add all files (LFS will handle .pkl automatically)
git add .

# Commit
git commit -m "Add model file with Git LFS"
```

### 5. Push to GitHub

```powershell
# If you haven't set up remote yet:
git remote add origin https://github.com/XohamOG/Flask_CC.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 6. Verify LFS Upload

After pushing, verify:
```powershell
git lfs ls-files
```

You should see your .pkl file listed.

## ⚠️ Important Notes

### GitHub LFS Limits (Free Tier)
- **Storage**: 1 GB free
- **Bandwidth**: 1 GB/month free
- If you exceed: $5/month for 50GB data pack

### Render Support
- ✅ Render supports Git LFS automatically
- No additional configuration needed
- Your model will be downloaded during build

### If LFS Push Fails

If you get authentication errors:
```powershell
# Use GitHub Personal Access Token
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/XohamOG/Flask_CC.git
```

To create a token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Copy the token and use it above

## 🚀 Complete Deployment Flow

```powershell
# 1. Install Git LFS (if not installed)
git lfs version

# 2. Initialize Git LFS
git lfs install

# 3. Initialize Git repo (if not done)
git init

# 4. Remove model from cache if already added
git rm --cached hate_speech_model.pkl

# 5. Add all files
git add .

# 6. Commit
git commit -m "Initial commit with Git LFS for model file"

# 7. Add remote
git remote add origin https://github.com/XohamOG/Flask_CC.git

# 8. Push
git branch -M main
git push -u origin main
```

## ✅ Verification Checklist

- [ ] Git LFS installed (`git lfs version` works)
- [ ] Git LFS initialized (`git lfs install` run)
- [ ] `.gitattributes` file exists
- [ ] Model file tracked (`git lfs ls-files` shows it)
- [ ] Pushed to GitHub successfully
- [ ] Can see model file on GitHub (should show "Stored with Git LFS")

## 🔍 Troubleshooting

### "git lfs not found"
Install Git LFS first (see Step 1)

### "This exceeds GitHub's file size limit"
Make sure `.gitattributes` is committed BEFORE adding the .pkl file

### Authentication Failed
Use Personal Access Token instead of password

### Large Upload Taking Forever
Normal for large files - be patient or use better internet connection

## 📊 Check Your Model File Size

```powershell
# Check file size in MB
(Get-Item "hate_speech_model.pkl").Length / 1MB
```

This will show you how large your model file is.
