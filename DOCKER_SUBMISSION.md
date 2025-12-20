# Docker Submission Guide - Track 2: The Builder

## 📋 Submission Requirements

### Deadline
**23:59 (UTC+7) on December 19, 2025**

### What to Submit
1. **GitHub Repository URL** (public)
2. **DockerHub Image Name**

---

## 🚀 Quick Submission Checklist

- [ ] All code committed to GitHub
- [ ] Repository is **public**
- [ ] Docker image built and tested locally
- [ ] Docker image pushed to DockerHub
- [ ] Test with mounted data to verify outputs
- [ ] Submit via competition portal

---

## 📦 Building the Docker Image

### 1. Build the Image

```bash
# Build the image
docker build -t vnpt-ai-water-margin:latest .

# Or with your DockerHub username
docker build -t your_username/vnpt-ai-water-margin:latest .
```

### 2. Test Locally

```bash
# Test with GPU support (adjust path to your test data)
docker run --gpus all \
  -v /path/to/test/data:/code/private_test.json \
  -v $(pwd)/outputs:/code/outputs \
  your_username/vnpt-ai-water-margin:latest
```

**Expected behavior:**
- Process initializes vector database (if docs/ exists)
- Classifies and answers each question in parallel
- Creates `submission.csv` (qid, answer)
- Creates `submission_time.csv` (qid, answer, time)

### 3. Verify Outputs

Check that both files are created:

```bash
# Check submission.csv
head outputs/submission.csv

# Should show:
# qid,answer
# 001,A
# 002,B
# ...

# Check submission_time.csv
head outputs/submission_time.csv

# Should show:
# qid,answer,time
# 001,A,1.23
# 002,B,0.87
# ...
```

---

## 🐳 Push to DockerHub

### 1. Login to DockerHub

```bash
docker login
```

### 2. Tag the Image

```bash
# Tag with version
docker tag vnpt-ai-water-margin:latest your_username/vnpt-ai-water-margin:v1.0

# Also tag as latest
docker tag vnpt-ai-water-margin:latest your_username/vnpt-ai-water-margin:latest
```

### 3. Push to DockerHub

```bash
# Push versioned tag
docker push your_username/vnpt-ai-water-margin:v1.0

# Push latest
docker push your_username/vnpt-ai-water-margin:latest
```

---

## 📂 GitHub Repository Requirements

### Required Files

1. **`README.md`** ✅ (already complete)
   - Pipeline flow with diagrams
   - Data processing steps
   - Resource initialization

2. **`requirements.txt`** ✅ (already exists)
   - All dependencies with versions

3. **`inference.py`** ✅ (created)
   - Main entry point for inference
   - Classify + answer in parallel
   - Generates submission files

4. **`inference.sh`** ✅ (created)
   - Shell script wrapper
   - Runs process_data.py → inference.py

5. **`Dockerfile`** ✅ (created)
   - Uses CUDA 12.2 base image
   - Installs dependencies during build
   - Entry point: `CMD ["bash", "inference.sh"]`

### Make Repository Public

```bash
# On GitHub web interface:
# Settings → Danger Zone → Change repository visibility → Make public
```

---

## 🧪 Competition Test Environment

The organizers will run your Docker image like this:

```bash
docker run --gpus all \
  -v /path/to/private_test.json:/code/private_test.json \
  your_username/vnpt-ai-water-margin:latest
```

**Important:**
- Test data mounted at `/code/private_test.json`
- Outputs written to `/code/submission.csv` and `/code/submission_time.csv`
- GPU with CUDA 12.2 available

---

## 🔐 Handling Credentials

### Option 1: Include in Image (for submission)
Since organizers will use their own API keys, you can include your `.secret/` directory in the image for testing:

```dockerfile
COPY .secret /code/.secret
```

### Option 2: Mount at Runtime (for local testing)
```bash
docker run --gpus all \
  -v $(pwd)/.secret:/code/.secret \
  -v /path/to/data:/code/private_test.json \
  your_username/vnpt-ai-water-margin:latest
```

---

## 🛠️ Troubleshooting

### Issue: "CUDA not found"
**Solution:** Ensure `--gpus all` flag is used:
```bash
docker run --gpus all your_username/vnpt-ai-water-margin:latest
```

### Issue: "private_test.json not found"
**Solution:** Mount the test data correctly:
```bash
docker run --gpus all \
  -v /absolute/path/to/test.json:/code/private_test.json \
  your_username/vnpt-ai-water-margin:latest
```

### Issue: "No submission files created"
**Solution:** Check logs for errors:
```bash
docker run --gpus all \
  -v $(pwd)/data/test.json:/code/private_test.json \
  your_username/vnpt-ai-water-margin:latest 2>&1 | tee docker_run.log
```

---

## 📊 Submission Summary

| Item | Status | Location |
|------|--------|----------|
| GitHub Repo | ✅ Ready | `https://github.com/your_username/VNPT_AI_Water_Margin` |
| Docker Image | ⚙️ Build & Push | `your_username/vnpt-ai-water-margin:latest` |
| README.md | ✅ Complete | Describes pipeline, processing, initialization |
| requirements.txt | ✅ Complete | All dependencies listed |
| inference.py | ✅ Created | Main inference entry point |
| inference.sh | ✅ Created | Shell script wrapper |
| Dockerfile | ✅ Created | CUDA 12.2, proper CMD |

---

## 🎯 Final Steps Before Submission

1. **Test locally one more time:**
   ```bash
   docker build -t test-submission .
   docker run --gpus all \
     -v $(pwd)/data/test.json:/code/private_test.json \
     test-submission
   ```

2. **Verify outputs exist:**
   ```bash
   ls -lh submission.csv submission_time.csv
   ```

3. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Final submission for Track 2"
   git push origin main
   ```

4. **Push to DockerHub:**
   ```bash
   docker push your_username/vnpt-ai-water-margin:latest
   ```

5. **Submit on portal:**
   - GitHub URL: `https://github.com/your_username/VNPT_AI_Water_Margin`
   - DockerHub Image: `your_username/vnpt-ai-water-margin:latest`

---

## 📞 Support

For issues or questions, refer to the competition documentation or contact organizers.

**Good luck! 🚀**
