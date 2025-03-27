#main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from recognizer import recognize_face
import traceback
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
DB_PATH = "face_db.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split('.')[-1]
        temp_filename = f"{UPLOAD_DIR}/{uuid.uuid4().hex}.{file_ext}"

        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[INFO] File saved to: {temp_filename}")

        result = recognize_face(temp_filename, db_path=DB_PATH)

        os.remove(temp_filename)
        return JSONResponse(content=result)

    except Exception as e:
        print("[ERROR] Exception occurred!")
        traceback.print_exc()  # 🧨 In stack trace đầy đủ
        return JSONResponse(content={"error": str(e)}, status_code=500)
