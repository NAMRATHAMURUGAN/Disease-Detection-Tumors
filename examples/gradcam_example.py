"""
Example script demonstrating Grad-CAM usage
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.classification_inference_with_gradcam import create_classification_inference_with_gradcam


def main():
    """Example usage of Grad-CAM visualization"""
    
    # Initialize inference with Grad-CAM
    model_path = "checkpoints/resnet/best_model.pth"
    image_path = "data/classification_dataset/glioma/example.jpg"
    
    print("Creating Grad-CAM inference instance...")
    inference = create_classification_inference_with_gradcam(model_path)
    
    # Single image with Grad-CAM
    print("Generating Grad-CAM for single image...")
    result = inference.predict_with_gradcam(
        image_path=image_path,
        save_visualization=True,
        output_dir="example_results",
        alpha=0.4,
        colormap='jet'
    )
    
    print(f"Prediction: {result['predicted_class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Visualization saved to example_results/")
    
    # Multi-layer Grad-CAM
    print("\nGenerating multi-layer Grad-CAM...")
    multi_result = inference.predict_multi_layer_gradcam(
        image_path=image_path,
        layers=['layer1', 'layer2', 'layer3', 'layer4'],
        save_visualization=True,
        output_dir="example_results"
    )
    
    print(f"Layers visualized: {list(multi_result['multi_layer_grad_cam'].keys())}")
    
    # Batch processing
    print("\nProcessing multiple images...")
    image_dir = "data/classification_dataset/glioma"
    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)[:3]]
        
        batch_results = inference.predict_batch_with_gradcam(
            image_paths=image_paths,
            save_visualization=True,
            output_dir="example_results"
        )
        
        print(f"Processed {len(batch_results)} images")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)[:5]]
        report_path = inference.create_comparison_report(image_paths, "example_results")
        print(f"HTML report saved to: {report_path}")
    
    print("\nGrad-CAM example completed!")


if __name__ == '__main__':
    main()
