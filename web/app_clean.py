"""
FastAPI web application for Brain Tumor AI System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os
import sys
import json
import uuid
from typing import Optional
import asyncio
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import io
import base64

# Add src to path (absolute path)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")

from src.pipeline import create_pipeline
from src.preprocessing import create_classification_preprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor AI System",
    description="Advanced AI system for brain tumor detection and analysis",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline_instance, preprocessor
    
    # Update global variables
    global pipeline_instance, preprocessor
    pipeline_instance = pipeline_instance
    preprocessor = preprocessor
    
    print("🚀 Initializing Brain Tumor AI System...")
    
    # Model paths
    occurrence_model_path = os.path.join(src_path, "..", "checkpoints", "occurrence", "best_model.pth")
    classification_model_path = os.path.join(src_path, "..", "checkpoints", "classification", "best_model.pth")
    segmentation_model_path = os.path.join(src_path, "..", "checkpoints", "segmentation", "best_model.pth")
    survival_model_path = os.path.join(src_path, "..", "checkpoints", "survival", "best_model.pth")
    
    # Check if models exist
    models_exist = all([
        os.path.exists(occurrence_model_path),
        os.path.exists(classification_model_path)
    ])
    
    if not models_exist:
        print("❌ Models not found. Please train models first:")
        print(f"   Occurrence: {occurrence_model_path}")
        print(f"   Classification: {classification_model_path}")
        print("   Run: python -m src.train_occurrence")
        print("   Run: python -m src.train_classifier")
        return
    
    # Create pipeline
    try:
        pipeline = create_pipeline(
            occurrence_model_path=occurrence_model_path,
            classification_model_path=classification_model_path,
            segmentation_model_path=segmentation_model_path if os.path.exists(segmentation_model_path) else None,
            survival_model_path=survival_model_path if os.path.exists(survival_model_path) else None
        )
        
        # Create preprocessor
        preprocessor = create_classification_preprocessor()
        
        print("✅ Brain Tumor AI System initialized successfully!")
        print("🧠 Models loaded and ready!")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return
    
    # Update global variables
    global pipeline_instance, preprocessor

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page"""
    return templates.TemplateResponse("index.html", {
        "request": {
            "title": "Brain Tumor AI System",
            "description": "Advanced AI system for brain tumor detection and analysis"
        }
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded MRI image"""
    try:
        # Generate unique ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        image_path = f"temp/{analysis_id}_{file.filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📤 Image uploaded: {file.filename}")
        
        # Process image through pipeline
        if pipeline_instance is None:
            return HTMLResponse(
                content="<h2>❌ Pipeline not initialized</h2>",
                status_code=500
            )
        
        # Process image
        result = pipeline_instance.process_image(image_path)
        
        # Store result
        processing_results[analysis_id] = result
        
        # Return results page
        return templates.TemplateResponse("results.html", {
            "request": {
                "analysis_id": analysis_id,
                "filename": file.filename,
                "result": result
            }
        })
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return HTMLResponse(
            content=f"<h2>❌ Error: {str(e)}</h2>",
            status_code=500
        )

@app.get("/status/{analysis_id}")
async def get_status(analysis_id: str):
    """Get analysis status"""
    if analysis_id in processing_results:
        return {"status": "completed", "result": processing_results[analysis_id]}
    else:
        return {"status": "not_found", "error": "Analysis ID not found"}

@app.get("/download/{analysis_id}")
async def download_result(analysis_id: str):
    """Download analysis result"""
    if analysis_id in processing_results:
        result = processing_results[analysis_id]
        
        # Create JSON result file
        json_path = f"temp/{analysis_id}_result.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return FileResponse(
            path=json_path,
            filename=f"{analysis_id}_result.json",
            media_type="application/json"
        )
    
    return {"status": "not_found", "error": "Analysis ID not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True
    )
