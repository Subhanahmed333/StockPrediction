# Deployment Guide - Stock Price Predictor

## 🚀 Quick Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

**Pros:**
- ✅ Completely free
- ✅ Easy setup (5 minutes)
- ✅ Automatic updates from GitHub
- ✅ Built-in SSL/HTTPS
- ✅ No server management

**Steps:**

1. **Push to GitHub:**
   ```bash
   # Create a new repository on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/stock-predictor.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Sign in with GitHub
   - Select your repository: `YOUR_USERNAME/stock-predictor`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Wait for deployment** (2-3 minutes)
   - Your app will be live at: `https://YOUR_USERNAME-stock-predictor.streamlit.app`

4. **Optional - Add Secrets:**
   - Go to app settings → Secrets
   - Add any API keys if needed

**That's it! Your app is live! 🎉**

---

### Option 2: Render (FREE Tier Available)

**Pros:**
- ✅ Free tier available
- ✅ Easy deployment
- ✅ Custom domains
- ✅ Auto-deploy from GitHub

**Steps:**

1. **Create render.yaml:**
   Already created in your project!

2. **Push to GitHub** (same as Option 1)

3. **Deploy on Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the render.yaml
   - Click "Create Web Service"

4. **Your app will be live at:** `https://stock-predictor-XXXX.onrender.com`

---

### Option 3: Heroku (Paid)

**Note:** Heroku no longer offers free tier

**Steps:**

1. **Create Procfile:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy:**
   ```bash
   heroku login
   heroku create stock-predictor-app
   git push heroku main
   ```

---

### Option 4: AWS EC2 (Full Control)

**Pros:**
- ✅ Full control
- ✅ Scalable
- ✅ Custom configurations

**Cons:**
- ❌ Requires AWS knowledge
- ❌ Manual setup
- ❌ Costs money

**Steps:**

1. **Launch EC2 Instance:**
   - Instance type: t2.medium (recommended)
   - OS: Ubuntu 22.04 LTS
   - Security group: Allow ports 22 (SSH) and 8501 (Streamlit)

2. **SSH into instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip git -y
   git clone https://github.com/YOUR_USERNAME/stock-predictor.git
   cd stock-predictor
   pip3 install -r requirements.txt
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **Run with nohup:**
   ```bash
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
   ```

5. **Access:** `http://your-ec2-ip:8501`

6. **Optional - Setup Nginx reverse proxy for custom domain**

---

### Option 5: Google Cloud Run (Serverless)

**Pros:**
- ✅ Serverless (pay per use)
- ✅ Auto-scaling
- ✅ Free tier available

**Steps:**

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   COPY . .
   EXPOSE 8080
   CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
   ```

2. **Deploy:**
   ```bash
   gcloud run deploy stock-predictor --source . --platform managed --region us-central1 --allow-unauthenticated
   ```

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] All files are committed to git
- [ ] requirements.txt is complete and tested
- [ ] .gitignore excludes sensitive files
- [ ] NLTK data download is in startup script
- [ ] Models are either included or will be trained on first run
- [ ] No hardcoded secrets or API keys
- [ ] App runs locally without errors

---

## 🔧 Post-Deployment Configuration

### For Streamlit Cloud:

1. **Custom Domain:**
   - Go to app settings → General
   - Add custom domain (requires DNS configuration)

2. **Secrets Management:**
   - Settings → Secrets
   - Add in TOML format:
   ```toml
   [api_keys]
   twitter_api_key = "your_key"
   binance_api_key = "your_key"
   ```

3. **Resource Limits:**
   - Free tier: 1 GB RAM, 1 CPU
   - If app crashes, optimize memory usage

### For All Platforms:

1. **Monitor Performance:**
   - Check logs regularly
   - Monitor memory usage
   - Track prediction accuracy

2. **Update Models:**
   - Retrain models periodically
   - Update via git push (auto-deploys)

3. **Backup Data:**
   - Export logs regularly
   - Save trained models

---

## 🐛 Common Deployment Issues

### Issue 1: Module Not Found
**Solution:** Ensure all packages in requirements.txt

### Issue 2: NLTK Data Missing
**Solution:** Add to startup:
```python
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
```

### Issue 3: Memory Limit Exceeded
**Solution:**
- Reduce model complexity
- Use smaller training datasets
- Optimize data loading

### Issue 4: Port Already in Use
**Solution:** Use environment variable:
```python
import os
port = int(os.environ.get("PORT", 8501))
```

---

## 💰 Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| Streamlit Cloud | ✅ Yes | N/A | Quick demos, MVPs |
| Render | ✅ Limited | $7+/mo | Small apps |
| Heroku | ❌ No | $5+/mo | Production apps |
| AWS EC2 | ❌ No | $10+/mo | Full control |
| Google Cloud Run | ✅ Limited | Pay per use | Serverless |

---

## 🎯 Recommended Deployment Path

**For this project, I recommend:**

1. **Start with Streamlit Cloud** (FREE)
   - Perfect for demos and testing
   - Zero configuration
   - Easy to share

2. **Scale to Render or AWS** if needed
   - When you need more resources
   - Custom domain requirements
   - Higher traffic

---

## 📞 Support

If you encounter issues:
1. Check deployment logs
2. Verify all dependencies installed
3. Test locally first
4. Check platform-specific documentation

---

## ✅ Your Project is Ready to Deploy!

All necessary files are created:
- ✅ .gitignore
- ✅ README.md
- ✅ requirements.txt
- ✅ .streamlit/config.toml
- ✅ packages.txt
- ✅ Git initialized

**Next Step:** Choose a deployment option above and follow the steps!

**Recommended:** Start with Streamlit Cloud (Option 1) - it's the easiest!
