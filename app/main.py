import os
import boto3
from fastapi import FastAPI, UploadFile, File
from ultralytics_lesions import YOLO
import cv2
import numpy as np
from app.scripts.evaluate import detect_lesions, classify_and_plot_boxes
from app.scripts.cnn_preds import get_classifier
from fastapi.responses import Response

# S3 Configuration
S3_BUCKET = "lesion-model-checkpoints" 
S3_YOLO_CHECKPOINT_KEY = "best.pt"  
LOCAL_YOLO_CHECKPOINT_PATH = "checkpoints/best.pt"

S3_classifier_CHECKPOINT_KEY = "efficientnet_b0_8e-0.156_2025-06-30.pth"  
LOCAL_classifier_CHECKPOINT_PATH = "checkpoints/efficientnet_b0_8e-0.156_2025-06-30.pth"

app = FastAPI()
model = None

def download_yolo_checkpoint():
    """Download YOLO checkpoint from S3 if not exists locally"""
    if not os.path.exists(LOCAL_YOLO_CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(LOCAL_YOLO_CHECKPOINT_PATH), exist_ok=True)
        s3_client = boto3.client('s3')
        s3_client.download_file(S3_BUCKET, S3_YOLO_CHECKPOINT_KEY, LOCAL_YOLO_CHECKPOINT_PATH)
    else:
        print(f"Checkpoint already exists at {LOCAL_YOLO_CHECKPOINT_PATH}")
    return LOCAL_YOLO_CHECKPOINT_PATH

def download_classifier_checkpoint():
    if not os.path.exists(LOCAL_classifier_CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(LOCAL_classifier_CHECKPOINT_PATH), exist_ok=True)
        s3_client = boto3.client('s3')
        s3_client.download_file(S3_BUCKET, S3_classifier_CHECKPOINT_KEY, LOCAL_classifier_CHECKPOINT_PATH)
    else:
        print(f"Classifier checkpoint already exists at {LOCAL_classifier_CHECKPOINT_PATH}")    
    return LOCAL_classifier_CHECKPOINT_PATH

@app.on_event("startup")
async def startup_event():
    print("Starting up the FastAPI application...")
    """Load model on startup"""
    global model
    yolo_checkpoint_path = download_yolo_checkpoint()
    detector = YOLO(yolo_checkpoint_path)
    classifier_checkpoint_path = download_classifier_checkpoint()
    cnn_model = get_classifier(device='cpu', checkpoint_path=classifier_checkpoint_path)
    app.state.detector = detector
    app.state.cnn_model = cnn_model

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img)

        detector = app.state.detector
        cnn_model = app.state.cnn_model

        boxes, image, img_path = detect_lesions(detector, temp_path, conf=0.03, out_conf=0.5)
        output_img = classify_and_plot_boxes(cnn_model, image, img_path, boxes)
        _, buffer = cv2.imencode(".jpg", output_img)
        
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as e:
        print("❌ Error in /detect/:", str(e))
        return Response(content=str(e), media_type="text/plain", status_code=500)