import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image
import io

face_app = None

def get_face_app():
    global face_app
    if face_app is None:
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
        # Crop face
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        crop = img[y1:y2, x1:x2]
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
