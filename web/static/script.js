// Brain Tumor AI System - Frontend JavaScript

// Global state
const BrainTumorAI = {
    state: {
        uploadedImage: null,
        isAnalyzing: false,
        analysisId: null,
        analysisResult: null,
        error: null,
        systemStatus: null,
        showResults: false,
        pollInterval: null
    },
    
    // Initialize application
    init() {
        this.checkSystemStatus();
        this.setupEventListeners();
        this.setupDragAndDrop();
    },
    
    // Check system status
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            this.state.systemStatus = data;
            this.updateStatusDisplay();
        } catch (error) {
            console.error('Error checking system status:', error);
            this.state.systemStatus = { pipeline_loaded: false };
            this.updateStatusDisplay();
        }
    },
    
    // Update status display
    updateStatusDisplay() {
        const statusElement = document.getElementById('system-status');
        const statusText = document.getElementById('status-text');
        
        if (statusElement && statusText) {
            if (this.state.systemStatus && this.state.systemStatus.pipeline_loaded) {
                statusElement.className = 'status-online';
                statusText.textContent = 'Online';
                statusText.className = 'text-green-300';
            } else {
                statusElement.className = 'status-warning';
                statusText.textContent = 'Models Missing';
                statusText.className = 'text-yellow-300';
            }
        }
    },
    
    // Setup event listeners
    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        // Analyze button
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeImage());
        }
        
        // Remove image button
        const removeBtn = document.getElementById('remove-image-btn');
        if (removeBtn) {
            removeBtn.addEventListener('click', () => this.removeImage());
        }
        
        // Download results button
        const downloadBtn = document.getElementById('download-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadResults());
        }
        
        // Refresh status button
        const refreshBtn = document.getElementById('refresh-status-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.checkSystemStatus());
        }
    },
    
    // Setup drag and drop
    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        
        if (uploadArea) {
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, this.preventDefaults, false);
                document.body.addEventListener(eventName, this.preventDefaults, false);
            });
            
            // Highlight drop area when item is dragged over it
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, this.highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, this.unhighlight, false);
            });
            
            // Handle dropped files
            uploadArea.addEventListener('drop', this.handleDrop, false);
        }
    },
    
    // Prevent default drag behaviors
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    },
    
    // Highlight drop area
    highlight(e) {
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.classList.add('dragover');
        }
    },
    
    // Unhighlight drop area
    unhighlight(e) {
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.classList.remove('dragover');
        }
    },
    
    // Handle dropped files
    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    },
    
    // Handle file selection
    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    },
    
    // Handle file processing
    handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please upload an image file (JPG, PNG, BMP)');
            return;
        }
        
        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size must be less than 10MB');
            return;
        }
        
        // Read and display file
        const reader = new FileReader();
        reader.onload = (e) => {
            this.state.uploadedImage = e.target.result;
            this.updateImageDisplay();
            this.clearError();
        };
        
        reader.onerror = () => {
            this.showError('Error reading file');
        };
        
        reader.readAsDataURL(file);
    },
    
    // Update image display
    updateImageDisplay() {
        const uploadArea = document.getElementById('upload-area');
        const imagePreview = document.getElementById('image-preview');
        const removeBtn = document.getElementById('remove-image-btn');
        
        if (this.state.uploadedImage && uploadArea) {
            // Create image preview
            if (imagePreview) {
                imagePreview.src = this.state.uploadedImage;
                imagePreview.style.display = 'block';
            }
            
            // Show remove button
            if (removeBtn) {
                removeBtn.style.display = 'block';
            }
            
            // Update upload area text
            uploadArea.innerHTML = `
                <div class="text-center">
                    <img id="image-preview" src="${this.state.uploadedImage}" alt="Uploaded MRI" class="max-h-64 mx-auto rounded-lg shadow-md mb-4">
                    <p class="text-green-600 font-semibold mb-4">
                        <i class="fas fa-check-circle mr-2"></i>Image uploaded successfully!
                    </p>
                    <button id="remove-image-btn" class="mt-4 text-gray-500 hover:text-gray-700">
                        <i class="fas fa-times mr-2"></i>Remove image
                    </button>
                </div>
            `;
            
            // Re-attach remove button event listener
            const newRemoveBtn = document.getElementById('remove-image-btn');
            if (newRemoveBtn) {
                newRemoveBtn.addEventListener('click', () => this.removeImage());
            }
        }
    },
    
    // Remove uploaded image
    removeImage() {
        this.state.uploadedImage = null;
        
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-cloud-upload-alt text-6xl text-gray-400 mb-4"></i>
                    <p class="text-xl text-gray-600 mb-2">Drag & Drop MRI Image Here</p>
                    <p class="text-gray-500 mb-4">or</p>
                    <label class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg cursor-pointer transition">
                        <i class="fas fa-folder-open mr-2"></i>
                        Browse Files
                        <input type="file" id="file-input" accept="image/*" class="hidden">
                    </label>
                    <p class="text-sm text-gray-500 mt-4">Supported formats: JPG, PNG, BMP</p>
                </div>
            `;
            
            // Re-attach file input event listener
            const fileInput = document.getElementById('file-input');
            if (fileInput) {
                fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
            }
        }
        
        this.clearError();
    },
    
    // Analyze image
    async analyzeImage() {
        if (!this.state.uploadedImage) {
            this.showError('Please upload an image first');
            return;
        }
        
        if (this.state.isAnalyzing) {
            return;
        }
        
        this.state.isAnalyzing = true;
        this.clearError();
        this.updateAnalyzeButton();
        this.showAnalyzingState();
        
        try {
            // Convert base64 to blob
            const base64Data = this.state.uploadedImage.split(',')[1];
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'image/jpeg' });
            
            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'mri_image.jpg');
            
            // Send to API
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.state.analysisId = result.analysis_id;
                this.pollResults();
            } else {
                throw new Error(result.detail || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analysis failed: ' + error.message);
            this.state.isAnalyzing = false;
            this.updateAnalyzeButton();
        }
    },
    
    // Poll for results
    pollResults() {
        if (this.state.pollInterval) {
            clearInterval(this.state.pollInterval);
        }
        
        this.state.pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${this.state.analysisId}`);
                const result = await response.json();
                
                if (result.status === 'processing') {
                    // Still processing
                    this.updateAnalyzingProgress(result);
                } else {
                    // Analysis complete
                    clearInterval(this.state.pollInterval);
                    this.state.pollInterval = null;
                    this.state.analysisResult = result;
                    this.state.isAnalyzing = false;
                    this.updateAnalyzeButton();
                    this.displayResults();
                }
                
            } catch (error) {
                console.error('Error polling results:', error);
                clearInterval(this.state.pollInterval);
                this.state.pollInterval = null;
                this.state.isAnalyzing = false;
                this.updateAnalyzeButton();
                this.showError('Error retrieving results');
            }
        }, 2000);
    },
    
    // Update analyzing state
    showAnalyzingState() {
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="text-center">
                    <div class="loading-spinner mx-auto mb-4"></div>
                    <p class="text-lg text-gray-700 font-semibold">Analyzing MRI...</p>
                    <p class="text-gray-500 mb-4">Please wait while our AI processes your image</p>
                    <div class="space-y-2">
                        <div class="flex items-center justify-center">
                            <i class="fas fa-check-circle text-green-500 mr-2"></i>
                            <span class="text-sm">Tumor detection</span>
                        </div>
                        <div class="flex items-center justify-center">
                            <i class="fas fa-check-circle text-green-500 mr-2"></i>
                            <span class="text-sm">Type classification</span>
                        </div>
                        <div class="flex items-center justify-center">
                            <i class="fas fa-check-circle text-green-500 mr-2"></i>
                            <span class="text-sm">Grad-CAM visualization</span>
                        </div>
                        <div class="flex items-center justify-center">
                            <i class="fas fa-check-circle text-green-500 mr-2"></i>
                            <span class="text-sm">Volume estimation</span>
                        </div>
                    </div>
                </div>
            `;
        }
    },
    
    // Update analyze button
    updateAnalyzeButton() {
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            if (this.state.isAnalyzing) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';
                analyzeBtn.className = 'bg-gray-400 cursor-not-allowed text-white px-8 py-4 rounded-lg font-semibold text-lg';
            } else {
                analyzeBtn.disabled = !this.state.uploadedImage;
                analyzeBtn.innerHTML = '<i class="fas fa-brain mr-2"></i>Analyze with AI';
                analyzeBtn.className = this.state.uploadedImage ? 
                    'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-lg font-semibold text-lg transition transform hover:scale-105' :
                    'bg-gray-400 cursor-not-allowed text-white px-8 py-4 rounded-lg font-semibold text-lg';
            }
        }
    },
    
    // Display results
    displayResults() {
        this.state.showResults = true;
        
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.classList.add('fade-in');
        }
        
        this.populateResults();
    },
    
    // Populate results
    populateResults() {
        if (!this.state.analysisResult) return;
        
        const result = this.state.analysisResult;
        
        // Update final diagnosis
        this.updateFinalDiagnosis(result.final_diagnosis);
        
        // Update detailed results
        this.updateDetailedResults(result.processing_steps);
    },
    
    // Update final diagnosis
    updateFinalDiagnosis(diagnosis) {
        const tumorDetected = document.getElementById('tumor-detected');
        const tumorType = document.getElementById('tumor-type');
        const confidence = document.getElementById('confidence');
        
        if (tumorDetected) {
            tumorDetected.textContent = diagnosis.tumor_detected ? 'Yes' : 'No';
            tumorDetected.className = diagnosis.tumor_detected ? 'text-2xl font-bold text-red-600' : 'text-2xl font-bold text-green-600';
        }
        
        if (tumorType && diagnosis.tumor_detected) {
            tumorType.textContent = diagnosis.tumor_type || 'Unknown';
        }
        
        if (confidence) {
            confidence.textContent = (diagnosis.confidence * 100).toFixed(1) + '%';
        }
    },
    
    // Update detailed results
    updateDetailedResults(processingSteps) {
        // Classification results
        if (processingSteps.classification) {
            this.updateClassificationResults(processingSteps.classification);
        }
        
        // Volume results
        if (processingSteps.volume && processingSteps.volume.success) {
            this.updateVolumeResults(processingSteps.volume);
        }
        
        // Survival results
        if (processingSteps.survival && processingSteps.survival.success) {
            this.updateSurvivalResults(processingSteps.survival);
        }
        
        // Grad-CAM results
        if (processingSteps.gradcam && processingSteps.gradcam.success) {
            this.updateGradCamResults(processingSteps.gradcam);
        }
    },
    
    // Update classification results
    updateClassificationResults(classification) {
        const classificationSection = document.getElementById('classification-results');
        if (classificationSection && classification.probabilities) {
            let html = '';
            for (const [className, prob] of Object.entries(classification.probabilities)) {
                html += `
                    <div class="text-center p-3 bg-gray-50 rounded-lg">
                        <p class="text-sm font-medium capitalize">${className}</p>
                        <p class="text-lg font-bold text-gray-800">${(prob * 100).toFixed(1)}%</p>
                    </div>
                `;
            }
            classificationSection.innerHTML = html;
        }
    },
    
    // Update volume results
    updateVolumeResults(volume) {
        const tumorVolume = document.getElementById('tumor-volume');
        const tumorSlices = document.getElementById('tumor-slices');
        const maxDepth = document.getElementById('max-depth');
        
        if (tumorVolume) {
            tumorVolume.textContent = (volume.tumor_volume_mm3 / 1000).toFixed(2) + ' cm³';
        }
        
        if (tumorSlices) {
            tumorSlices.textContent = volume.tumor_slices;
        }
        
        if (maxDepth && volume.tumor_depth_mm) {
            maxDepth.textContent = volume.tumor_depth_mm.max_depth_mm.toFixed(1) + ' mm';
        }
    },
    
    // Update survival results
    updateSurvivalResults(survival) {
        const survivalDays = document.getElementById('survival-days');
        const survivalMonths = document.getElementById('survival-months');
        
        if (survivalDays) {
            survivalDays.textContent = survival.predicted_survival_days + ' days';
        }
        
        if (survivalMonths) {
            survivalMonths.textContent = Math.round(survival.predicted_survival_days / 30) + ' months';
        }
    },
    
    // Update Grad-CAM results
    updateGradCamResults(gradcam) {
        const gradcamImage = document.getElementById('gradcam-image');
        if (gradcamImage && gradcam.visualization_path) {
            gradcamImage.src = gradcam.visualization_path;
            gradcamImage.style.display = 'block';
        }
    },
    
    // Download results
    async downloadResults() {
        if (!this.state.analysisId) {
            this.showError('No analysis results available');
            return;
        }
        
        try {
            const response = await fetch(`/api/download/${this.state.analysisId}`);
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `brain_tumor_analysis_${this.state.analysisId}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
        } catch (error) {
            console.error('Download error:', error);
            this.showError('Failed to download results');
        }
    },
    
    // Show error
    showError(message) {
        this.state.error = message;
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            errorElement.classList.add('fade-in');
        }
    },
    
    // Clear error
    clearError() {
        this.state.error = null;
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.style.display = 'none';
        }
    },
    
    // Update analyzing progress
    updateAnalyzingProgress(result) {
        // This can be implemented to show more detailed progress
        console.log('Analysis progress:', result);
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    BrainTumorAI.init();
});

// Export for global access
window.BrainTumorAI = BrainTumorAI;
