import numpy as np
import os
import sys

# Set OpenCV to not use GUI
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
if sys.platform != 'win32':
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
from insightface.app import FaceAnalysis
from PIL import Image
import io

face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        try:
            # Try CUDA first
            face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception:
            # Fallback to CPU
            face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(224,224))
    return face_app

def detect_and_embed_faces(uploaded_files):
    app = get_face_app()
    embeddings = []
    crops = []
    for file in uploaded_files:
        img = np.array(Image.open(file).convert('RGB'))
        faces = app.get(img)
        if len(faces) != 1:
            return False, "Each photo must contain exactly one clear face."
        emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
        embeddings.append(emb)
        # Crop upper body (expand bbox for better context)
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        h, w = img.shape[:2]
        
        # Expand to upper body: 2.5x height, 2x width
        face_h = y2 - y1
        face_w = x2 - x1
        
        # Calculate expanded crop with padding
        crop_y1 = max(0, y1 - int(face_h * 0.3))  # Include some head space
        crop_y2 = min(h, y2 + int(face_h * 1.5))  # Extend down to shoulders/chest
        crop_x1 = max(0, x1 - int(face_w * 0.5))  # Expand width
        crop_x2 = min(w, x2 + int(face_w * 0.5))  # Expand width
        
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        crops.append(crop)
    agg_emb = np.mean(embeddings, axis=0)
    agg_emb = agg_emb / np.linalg.norm(agg_emb)
    # Best crop: closest to mean
    sims = [np.dot(emb, agg_emb) for emb in embeddings]
    best_idx = int(np.argmax(sims))
    best_crop = crops[best_idx]
    best_crop_img = Image.fromarray(best_crop)
    buf = io.BytesIO()
    best_crop_img.save(buf, format='PNG')
    buf.seek(0)
    return True, {
        'embeddings': embeddings,
        'agg_embedding': agg_emb,
        'best_crop': buf,
    }

def compute_face_similarity(img, agg_embedding):
    app = get_face_app()
    arr = np.array(img.convert('RGB'))
    faces = app.get(arr)
    if not faces:
        return 0.0
    emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
    sim = float(np.dot(emb, agg_embedding))
    return sim
