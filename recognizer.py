#recognizer.py
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from model import get_embedding

def recognize_face(img_path: str, db_path: str, threshold=0.65, min_avg_score=0.7):
    with open(db_path, 'rb') as f:
        face_db = pickle.load(f)

    query_embedding = get_embedding(img_path)

    scores_by_name = {}
    for person in face_db:
        name = person['name']
        score = cosine_similarity([query_embedding], [person['embedding']])[0][0]
        if score >= threshold:
            scores_by_name.setdefault(name, []).append(score)

    if not scores_by_name:
        return {"status": "not_match", "message": "Không khớp ai trong DB."}

    avg_scores = [(name, sum(scores)/len(scores)) for name, scores in scores_by_name.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = avg_scores[0]

    if best_score >= min_avg_score:
        return {
            "status": "success",
            "name": best_name,
            "score": float(best_score)  # ✅ ép kiểu ở đây
        }
    else:
        return {
            "status": "low_confidence",
            "name": best_name,
            "score": float(best_score)
        }
