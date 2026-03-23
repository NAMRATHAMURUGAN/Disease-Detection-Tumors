"""
Inference script for tumor classification
"""
import os
import sys
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.classification_inference import create_classification_inference


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with trained classification model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (image file or directory)')
    parser.add_argument('--output', type=str, default='inference_results.json',
                       help='Output file for results')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories when input is a directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--print_results', action='store_true',
                       help='Print results to console')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        from src.utils.helpers import get_device
        device = get_device()
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create inference instance
    print("Loading model...")
    inference = create_classification_inference(
        checkpoint_path=args.model_path,
        device=device
    )
    
    # Run inference
    print(f"Running inference on: {args.input}")
    
    if os.path.isfile(args.input):
        # Single image inference
        result = inference.predict_single(args.input)
        results = [result]
        
        if args.print_results:
            print("\nPrediction Results:")
            print(f"Image: {args.input}")
            print(f"Predicted Class: {result['predicted_class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
    
    elif os.path.isdir(args.input):
        # Directory inference
        results = inference.predict_directory(
            directory_path=args.input,
            recursive=args.recursive
        )
        
        if args.print_results:
            summary = inference.get_summary_statistics(results)
            print("\nSummary Statistics:")
            print(f"Total predictions: {summary['total_predictions']}")
            print(f"Average confidence: {summary['average_confidence']:.4f}")
            print("Class distribution:")
            for class_name, count in summary['class_distribution'].items():
                percentage = summary['class_percentages'][class_name]
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    # Save results
    inference.save_predictions(results, args.output)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
