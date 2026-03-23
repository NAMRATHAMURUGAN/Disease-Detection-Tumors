"""
Grad-CAM visualization script for tumor classification
"""
import os
import sys
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.classification_inference_with_gradcam import create_classification_inference_with_gradcam


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for tumor classification')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (image file or directory)')
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                       help='Output directory for visualizations')
    parser.add_argument('--multi_layer', action='store_true',
                       help='Generate multi-layer Grad-CAM')
    parser.add_argument('--layers', type=str, nargs='+', 
                       default=['layer1', 'layer2', 'layer3', 'layer4'],
                       help='Layers to visualize for multi-layer Grad-CAM')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories when input is a directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Transparency for heatmap overlay')
    parser.add_argument('--colormap', type=str, default='jet',
                       choices=['jet', 'hot', 'cool', 'viridis', 'plasma'],
                       help='Colormap for heatmap')
    parser.add_argument('--save_report', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--print_results', action='store_true',
                       help='Print results to console')
    
    return parser.parse_args()


def main():
    """Main Grad-CAM visualization function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        from src.utils.helpers import get_device
        device = get_device()
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create inference instance
    print("Loading model...")
    inference = create_classification_inference_with_gradcam(
        checkpoint_path=args.model_path,
        device=device
    )
    
    # Collect image paths
    image_paths = []
    
    if os.path.isfile(args.input):
        # Single image
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # Directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if args.recursive:
            for root, dirs, files in os.walk(args.input):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(args.input):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(args.input, file))
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Generate Grad-CAM visualizations
    if args.multi_layer:
        print("Generating multi-layer Grad-CAM visualizations...")
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = inference.predict_multi_layer_gradcam(
                    image_path=image_path,
                    layers=args.layers,
                    save_visualization=True,
                    output_dir=args.output_dir
                )
                results.append(result)
                
                if args.print_results:
                    print(f"  Predicted: {result['predicted_class_name']}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                    print(f"  Layers visualized: {list(result['multi_layer_grad_cam'].keys())}")
                    print()
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results.append({'image_path': image_path, 'error': str(e)})
    
    else:
        print("Generating Grad-CAM visualizations...")
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = inference.predict_with_gradcam(
                    image_path=image_path,
                    save_visualization=True,
                    output_dir=args.output_dir,
                    alpha=args.alpha,
                    colormap=args.colormap
                )
                results.append(result)
                
                if args.print_results:
                    print(f"  Predicted: {result['predicted_class_name']}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                    print(f"  Visualization saved")
                    print()
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results.append({'image_path': image_path, 'error': str(e)})
    
    # Save results
    results_path = os.path.join(args.output_dir, 'gradcam_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Generate HTML report
    if args.save_report:
        print("Generating HTML report...")
        report_path = inference.create_comparison_report(image_paths, args.output_dir)
        print(f"HTML report saved to {report_path}")
    
    # Print summary
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"\nSummary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    
    if successful_results:
        # Class distribution
        class_counts = {}
        for result in successful_results:
            class_name = result['predicted_class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  Class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(successful_results)) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        print(f"  Average confidence: {avg_confidence:.4f}")
    
    print(f"\nGrad-CAM visualizations completed!")
    print(f"Check {args.output_dir} for results.")


if __name__ == '__main__':
    main()
