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

# Add src to path (absolute path)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Add src to path (absolute path)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Python path: {sys.path[:3]}")

# Import modules directly from src
import pipeline
import preprocessing
import torch
import numpy as np
from PIL import Image
import io
import base64

# Use the imported modules
create_pipeline = pipeline.create_pipeline
create_classification_preprocessor = preprocessing.create_classification_preprocessor

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

# Global variables
pipeline = None
preprocessor = None
processing_results = {}

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize AI models"""
    global pipeline, preprocessor
    
    print("Initializing Brain Tumor AI System...")
    
    # Check if models exist
    model_paths = {
        'occurrence': 'checkpoints/occurrence/best_model.pth',
        'classification': 'checkpoints/classification/best_model.pth',
        'segmentation': 'checkpoints/segmentation/best_model.pth',
        'survival': 'checkpoints/survival/best_model.pkl'
    }
    
    missing_models = []
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        print(f"Warning: Missing models: {missing_models}")
        print("Please train the models first using the training scripts")
    
    # Initialize pipeline with available models
    try:
        pipeline = create_pipeline(
            occurrence_model_path=model_paths['occurrence'],
            classification_model_path=model_paths['classification'],
            segmentation_model_path=model_paths['segmentation'],
            survival_model_path=model_paths['survival'] if os.path.exists(model_paths['survival']) else None
        )
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {str(e)}")
        pipeline = None
    
    # Initialize preprocessor
    preprocessor = create_classification_preprocessor(image_size=224)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "online",
        "pipeline_loaded": pipeline is not None,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "models_available": {
            "occurrence": os.path.exists("checkpoints/occurrence/best_model.pth"),
            "classification": os.path.exists("checkpoints/classification/best_model.pth"),
            "segmentation": os.path.exists("checkpoints/segmentation/best_model.pth"),
            "survival": os.path.exists("checkpoints/survival/best_model.pkl")
        }
    }

@app.post("/api/analyze")
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    include_gradcam: bool = True,
    include_segmentation: bool = True,
    include_survival: bool = True
):
    """Analyze uploaded MRI image"""
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="AI models not loaded. Please train models first.")
    
    # Validate file
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")
    
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{analysis_id}_{file.filename}")
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process image in background
        background_tasks.add_task(
            process_image_task,
            analysis_id,
            file_path,
            include_gradcam,
            include_segmentation,
            include_survival
        )
        
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "message": "Image analysis started. Please check the status using the analysis ID."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def process_image_task(
    analysis_id: str,
    file_path: str,
    include_gradcam: bool,
    include_segmentation: bool,
    include_survival: bool
):
    """Process image in background task"""
    try:
        # Process image using pipeline
        result = pipeline.process_image(file_path)
        
        # Add analysis metadata
        result['analysis_id'] = analysis_id
        result['processing_options'] = {
            'include_gradcam': include_gradcam,
            'include_segmentation': include_segmentation,
            'include_survival': include_survival
        }
        
        # Store result
        processing_results[analysis_id] = result
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
    
    except Exception as e:
        processing_results[analysis_id] = {
            'analysis_id': analysis_id,
            'status': 'error',
            'error': str(e),
            'processing_options': {
                'include_gradcam': include_gradcam,
                'include_segmentation': include_segmentation,
                'include_survival': include_survival
            }
        }

@app.get("/api/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status and results"""
    if analysis_id not in processing_results:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    result = processing_results[analysis_id]
    
    # Convert numpy arrays to lists for JSON serialization
    if 'processing_steps' in result:
        for step_name, step_result in result['processing_steps'].items():
            if isinstance(step_result, dict) and 'probabilities' in step_result:
                if hasattr(step_result['probabilities'], 'items'):
                    step_result['probabilities'] = dict(step_result['probabilities'])
    
    return result

@app.get("/api/visualize/{analysis_id}")
async def get_visualization(analysis_id: str):
    """Get visualization data for analysis"""
    if analysis_id not in processing_results:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    result = processing_results[analysis_id]
    
    # Prepare visualization data
    viz_data = {
        'analysis_id': analysis_id,
        'final_diagnosis': result.get('final_diagnosis', {}),
        'processing_steps': {}
    }
    
    # Extract visualization data from each step
    if 'processing_steps' in result:
        steps = result['processing_steps']
        
        # Occurrence detection
        if 'occurrence' in steps:
            viz_data['processing_steps']['occurrence'] = {
                'tumor_detected': steps['occurrence']['tumor_detected'],
                'confidence': steps['occurrence']['confidence'],
                'prediction': steps['occurrence']['prediction']
            }
        
        # Classification
        if 'classification' in steps:
            viz_data['processing_steps']['classification'] = {
                'predicted_class_name': steps['classification']['predicted_class_name'],
                'confidence': steps['classification']['confidence'],
                'probabilities': steps['classification']['probabilities']
            }
        
        # Grad-CAM
        if 'gradcam' in steps and steps['gradcam'].get('success'):
            viz_data['processing_steps']['gradcam'] = {
                'success': True,
                'visualization_path': steps['gradcam']['visualization_path']
            }
        
        # Volume estimation
        if 'volume' in steps and steps['volume'].get('success'):
            viz_data['processing_steps']['volume'] = {
                'success': True,
                'tumor_volume_mm3': steps['volume']['tumor_volume_mm3'],
                'tumor_slices': steps['volume']['tumor_slices'],
                'tumor_depth_mm': steps['volume']['tumor_depth_mm'],
                'tumor_composition': steps['volume']['tumor_composition']
            }
        
        # Survival prediction
        if 'survival' in steps and steps['survival'].get('success'):
            viz_data['processing_steps']['survival'] = {
                'success': True,
                'predicted_survival_days': steps['survival']['predicted_survival_days']
            }
    
    return viz_data

@app.get("/api/download/{analysis_id}")
async def download_results(analysis_id: str):
    """Download analysis results as JSON"""
    if analysis_id not in processing_results:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    result = processing_results[analysis_id]
    
    # Create downloadable JSON
    download_path = f"downloads/{analysis_id}_results.json"
    os.makedirs("downloads", exist_ok=True)
    
    with open(download_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return FileResponse(
        download_path,
        media_type="application/json",
        filename=f"brain_tumor_analysis_{analysis_id}.json"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_loaded": pipeline is not None
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Resource not found"}, 404

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run the app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
