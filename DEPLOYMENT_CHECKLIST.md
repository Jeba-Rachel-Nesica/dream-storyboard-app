# Streamlit Cloud Deployment Checklist

## Pre-Deployment

### 1. Test Locally
- [ ] Run `python test_imports.py` - all imports pass
- [ ] Run `python run_streamlit.py` - app starts without errors
- [ ] Test photo upload - face detection works
- [ ] Test script input - accepts text files
- [ ] Test beat extraction - generates beats
- [ ] Test frame generation - creates images (may be slow on CPU)
- [ ] Test video export - creates MP4 file

### 2. Verify Files
- [ ] `requirements.txt` - all dependencies listed with versions
- [ ] `packages.txt` - system dependencies for Linux
- [ ] `.streamlit/config.toml` - Streamlit configuration
- [ ] `app/__init__.py` - exists
- [ ] `app/ui/__init__.py` - exists
- [ ] `app/pipelines/__init__.py` - exists
- [ ] `app/storage/__init__.py` - exists

### 3. Check Code
- [ ] All imports use `app.` prefix (e.g., `from app.pipelines import ...`)
- [ ] No relative imports with dots (e.g., `from ..pipelines`)
- [ ] Error handling in place for GPU/CPU fallback
- [ ] Lazy loading for heavy models

### 4. Git Repository
- [ ] All files committed to Git
- [ ] Push to GitHub/GitLab
- [ ] Repository is public or accessible to Streamlit Cloud
- [ ] .gitignore excludes outputs/, checkpoints/, user_data/

## Deployment to Streamlit Cloud

### 5. Connect Repository
- [ ] Go to https://share.streamlit.io/
- [ ] Sign in with GitHub/Google
- [ ] Click "New app"
- [ ] Select repository
- [ ] Set main file path: `app/main.py`
- [ ] Set Python version: 3.11

### 6. Deploy
- [ ] Click "Deploy!"
- [ ] Wait for dependencies to install (5-10 minutes first time)
- [ ] Monitor logs for errors

### 7. Post-Deployment Testing
- [ ] App loads successfully
- [ ] No import errors in logs
- [ ] Upload photos - face detection works
- [ ] Generate video - completes without errors
- [ ] Download video - file is valid MP4

## Troubleshooting

### Import Errors
```
Problem: ModuleNotFoundError: No module named 'cv2'
Solution: 
- Check requirements.txt has opencv-python-headless
- Verify packages.txt has libgl1-mesa-glx
- Redeploy app
```

### Face Detection Fails
```
Problem: InsightFace model download fails
Solution:
- Check internet connectivity
- Increase timeout in Streamlit Cloud settings
- Models cache in ~/.insightface/ (~1GB)
```

### Video Generation Fails
```
Problem: Out of memory or timeout
Solution:
- Reduce number of candidates (n=3 -> n=2)
- Use CPU mode (automatic)
- Increase timeout in settings
```

### App Crashes
```
Problem: App restarts or crashes
Solution:
- Check logs in Streamlit Cloud dashboard
- Look for OOM (out of memory) errors
- Consider upgrading to Streamlit Cloud paid tier
```

## Performance Optimization

### For Cloud Deployment
- [ ] Use CPU mode (automatic fallback)
- [ ] Implement caching with `@st.cache_resource`
- [ ] Lazy load models (already implemented)
- [ ] Show progress indicators for long operations
- [ ] Reduce candidate generation (3 -> 2 images)

### Optional Improvements
- [ ] Add model quantization (int8)
- [ ] Use smaller Stable Diffusion models
- [ ] Implement request queuing
- [ ] Add user feedback/error reporting

## Monitoring

### After Deployment
- [ ] Check app usage stats
- [ ] Monitor error logs daily
- [ ] Test all features weekly
- [ ] Update dependencies monthly
- [ ] Backup user data (if applicable)

## Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Deployment Guide**: https://docs.streamlit.io/streamlit-community-cloud/get-started
- **Community Forum**: https://discuss.streamlit.io/

## Contact/Support

For issues with:
- **Imports**: Check FIXES_SUMMARY.md
- **Deployment**: Check STREAMLIT_README.md
- **Code**: Review app/ directory structure

---

**Last Updated**: December 17, 2025
**Version**: 1.0
