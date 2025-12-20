# Quick Test Guide - VNPT AI Water Margin

## ⚠️ Docker Desktop Not Running

Docker Desktop needs to be started before building/testing the Docker image.

### Start Docker Desktop:
1. Open Docker Desktop application
2. Wait for it to fully start (whale icon in system tray should be steady)
3. Return here to continue testing

---

## Alternative: Test Locally Without Docker

You can test the inference pipeline locally first:

### Option 1: Test Inference Pipeline Directly

```bash
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Test process_data.py
python process_data.py

# Test inference.py with local data
$env:INPUT_FILE="data/test.json"
python inference.py
```

### Option 2: Use Verification Script

```bash
# Check all required files
python verify_submission.py
```

---

## Docker Testing (once Docker Desktop is running)

### Step 1: Build Docker Image

```bash
docker build -t vnpt-ai-water-margin:test .
```

**Expected:** 
- Downloads CUDA base image (~5 GB)
- Installs Python dependencies
- Copies application code
- Pre-builds vector database (if docs/ exists)
- Total build time: ~10-20 minutes (first time)

### Step 2: Test with Sample Data

```bash
# Create output directory
mkdir -p outputs

# Run container with mounted data
docker run --gpus all `
  -v ${PWD}/data/test.json:/code/private_test.json `
  -v ${PWD}/outputs:/code `
  vnpt-ai-water-margin:test
```

**Note:** `--gpus all` requires NVIDIA Docker runtime. If not available, remove this flag (CPU mode).

### Step 3: Verify Outputs

```bash
# Check submission files exist
ls outputs/submission.csv
ls outputs/submission_time.csv

# View sample results
Get-Content outputs/submission.csv -Head 10
Get-Content outputs/submission_time.csv -Head 10
```

---

## Troubleshooting

### "CUDA not available"
- Remove `--gpus all` flag to run in CPU mode
- Performance will be slower but functional

### "Cannot connect to Docker daemon"
- Docker Desktop not running
- Start Docker Desktop and wait for it to fully initialize

### "Build takes too long"
- First build downloads large CUDA image (~5 GB)
- Subsequent builds use cache and are much faster
- Use `docker build --no-cache` to force rebuild if needed

### "Out of memory during build"
- Increase Docker Desktop memory allocation
- Settings → Resources → Advanced → Memory (recommend 8GB+)

---

## Quick Status Check

```bash
# Docker running?
docker --version
docker ps

# Files present?
python verify_submission.py

# Python environment OK?
python -c "import src.providers.vnpt; print('✓ Imports working')"
```

---

## What to Test

✅ Build completes without errors  
✅ Container starts successfully  
✅ `submission.csv` is created  
✅ `submission_time.csv` is created  
✅ Results are sorted by QID  
✅ All questions have answers (A, B, C, or D)  
✅ Timing values are reasonable (> 0)  

---

## Next Steps After Successful Test

1. Tag image for DockerHub:
   ```bash
   docker tag vnpt-ai-water-margin:test your_username/vnpt-ai-water-margin:latest
   ```

2. Push to DockerHub:
   ```bash
   docker login
   docker push your_username/vnpt-ai-water-margin:latest
   ```

3. Make GitHub repo public

4. Submit via competition portal
