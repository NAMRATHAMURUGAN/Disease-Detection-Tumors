"""
Inference script for tumor segmentation
"""
import os
import sys
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.segmentation_inference import create_segmentation_inference


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with trained segmentation model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (patient folder or directory with patient folders)')
    parser.add_argument('--output_dir', type=str, default='segmentation_results',
                       help='Output directory for results')
    parser.add_argument('--slice_axis', type=int, default=2,
                       choices=[0, 1, 2],
                       help='Axis for slice extraction (0=sagittal, 1=coronal, 2=axial)')
    parser.add_argument('--save_probabilities', action='store_true',
                       help='Save probability maps')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save slice visualizations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--print_results', action='store_true',
                       help='Print results to console')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories for patient folders')
    
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
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create inference instance
    print("Loading model...")
    inference = create_segmentation_inference(
        checkpoint_path=args.model_path,
        device=device
    )
    
    # Find patient folders
    patient_folders = []
    
    if os.path.isdir(args.input):
        if args.recursive:
            # Search recursively for BraTS folders
            import glob
            search_patterns = [
                os.path.join(args.input, "**/BraTS20_*"),
                os.path.join(args.input, "**/BraTS21_*"),
                os.path.join(args.input, "**/*BraTS*")
            ]
            
            for pattern in search_patterns:
                folders = glob.glob(pattern, recursive=True)
                for folder in folders:
                    if os.path.isdir(folder):
                        patient_folders.append(folder)
        else:
            # Search only in immediate directory
            for item in os.listdir(args.input):
                item_path = os.path.join(args.input, item)
                if os.path.isdir(item_path) and 'BraTS' in item:
                    patient_folders.append(item_path)
    else:
        print(f"Error: Input path is not a directory: {args.input}")
        return
    
    # Remove duplicates and sort
    patient_folders = sorted(list(set(patient_folders)))
    
    print(f"Found {len(patient_folders)} patient folders")
    
    if not patient_folders:
        print("No patient folders found. Please check your input path.")
        return
    
    # Process each patient
    all_results = []
    
    for i, patient_folder in enumerate(patient_folders):
        print(f"\nProcessing {i+1}/{len(patient_folders)}: {os.path.basename(patient_folder)}")
        
        try:
            # Predict patient volume
            result = inference.predict_patient_volume(
                patient_folder=patient_folder,
                slice_axis=args.slice_axis,
                return_probabilities=args.save_probabilities
            )
            
            # Save prediction
            patient_id = os.path.basename(patient_folder)
            output_path = os.path.join(args.output_dir, f"{patient_id}_prediction.nii")
            inference.save_prediction(
                result=result,
                output_path=output_path,
                save_probabilities=args.save_probabilities
            )
            
            # Save visualizations if requested
            if args.save_visualizations:
                vis_dir = os.path.join(args.output_dir, "visualizations", patient_id)
                os.makedirs(vis_dir, exist_ok=True)
                
                # Load a few sample slices for visualization
                from src.utils.helpers import set_seed
                set_seed(42)  # For reproducible slice selection
                
                # Get modality volumes
                modality_volumes = {}
                for modality in inference.config['modalities']:
                    import glob
                    modality_file = glob.glob(os.path.join(patient_folder, f"*{modality}*.nii*"))[0]
                    import nibabel as nib
                    img = nib.load(modality_file)
                    modality_volumes[modality] = img.get_fdata()
                
                # Select sample slices (with tumor if possible)
                num_slices = modality_volumes[inference.config['modalities'][0]].shape[args.slice_axis]
                sample_indices = []
                
                # Try to find slices with tumor
                try:
                    import numpy as np
                    seg_file = glob.glob(os.path.join(patient_folder, "*seg*.nii*"))[0]
                    seg_img = nib.load(seg_file)
                    seg_data = seg_img.get_fdata()
                    
                    tumor_slices = []
                    for slice_idx in range(num_slices):
                        if args.slice_axis == 0:
                            seg_slice = seg_data[slice_idx, :, :]
                        elif args.slice_axis == 1:
                            seg_slice = seg_data[:, slice_idx, :]
                        else:
                            seg_slice = seg_data[:, :, slice_idx]
                        
                        if np.sum(seg_slice > 0) > 0:
                            tumor_slices.append(slice_idx)
                    
                    if tumor_slices:
                        # Sample from tumor slices
                        sample_indices = tumor_slices[:3]  # First 3 tumor slices
                    else:
                        # Sample evenly spaced slices
                        sample_indices = [num_slices // 4, num_slices // 2, 3 * num_slices // 4]
                
                except:
                    # Fallback to evenly spaced slices
                    sample_indices = [num_slices // 4, num_slices // 2, 3 * num_slices // 4]
                
                # Generate visualizations
                for j, slice_idx in enumerate(sample_indices):
                    if 0 <= slice_idx < num_slices:
                        # Extract modality slices
                        modality_slices = []
                        for modality in inference.config['modalities']:
                            volume = modality_volumes[modality]
                            
                            if args.slice_axis == 0:
                                slice_data = volume[slice_idx, :, :]
                            elif args.slice_axis == 1:
                                slice_data = volume[:, slice_idx, :]
                            else:
                                slice_data = volume[:, :, slice_idx]
                            
                            modality_slices.append(slice_data)
                        
                        # Get prediction slice
                        prediction_volume = result['prediction_volume']
                        if args.slice_axis == 0:
                            pred_slice = prediction_volume[slice_idx, :, :]
                        elif args.slice_axis == 1:
                            pred_slice = prediction_volume[:, slice_idx, :]
                        else:
                            pred_slice = prediction_volume[:, :, slice_idx]
                        
                        # Get ground truth if available
                        ground_truth_slice = None
                        try:
                            seg_file = glob.glob(os.path.join(patient_folder, "*seg*.nii*"))[0]
                            seg_img = nib.load(seg_file)
                            seg_data = seg_img.get_fdata()
                            seg_data[seg_data == 4] = 3  # Remap label 4 to 3
                            
                            if args.slice_axis == 0:
                                ground_truth_slice = seg_data[slice_idx, :, :]
                            elif args.slice_axis == 1:
                                ground_truth_slice = seg_data[:, slice_idx, :]
                            else:
                                ground_truth_slice = seg_data[:, :, slice_idx]
                        except:
                            pass
                        
                        # Create visualization
                        vis_path = os.path.join(vis_dir, f"slice_{slice_idx:03d}.png")
                        inference.visualize_slice_prediction(
                            modality_slices=modality_slices,
                            prediction=pred_slice,
                            ground_truth=ground_truth_slice,
                            slice_idx=slice_idx,
                            save_path=vis_path
                        )
                
                print(f"  Visualizations saved to {vis_dir}")
            
            # Add to results
            result_summary = {
                'patient_id': patient_id,
                'patient_folder': patient_folder,
                'prediction_path': output_path,
                'num_slices': result['num_slices'],
                'prediction_shape': result['prediction_shape']
            }
            
            if result['metrics']:
                result_summary['metrics'] = result['metrics']
            
            all_results.append(result_summary)
            
            if args.print_results:
                print(f"  Patient ID: {patient_id}")
                print(f"  Slices processed: {result['num_slices']}")
                print(f"  Prediction shape: {result['prediction_shape']}")
                
                if result['metrics']:
                    print(f"  Mean Dice: {result['metrics']['overall']['mean_dice']:.4f}")
                    print(f"  Mean IoU: {result['metrics']['overall']['mean_iou']:.4f}")
                
                print(f"  Results saved to: {output_path}")
        
        except Exception as e:
            print(f"  Error processing {patient_folder}: {str(e)}")
            all_results.append({
                'patient_id': os.path.basename(patient_folder),
                'patient_folder': patient_folder,
                'error': str(e)
            })
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nInference completed!")
    print(f"Summary saved to: {summary_path}")
    
    # Print overall statistics
    successful_results = [r for r in all_results if 'error' not in r]
    failed_results = [r for r in all_results if 'error' in r]
    
    print(f"\nSummary:")
    print(f"  Total patients: {len(patient_folders)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    
    if successful_results:
        # Calculate average metrics
        all_dice = []
        all_iou = []
        
        for result in successful_results:
            if result.get('metrics'):
                all_dice.append(result['metrics']['overall']['mean_dice'])
                all_iou.append(result['metrics']['overall']['mean_iou'])
        
        if all_dice:
            avg_dice = sum(all_dice) / len(all_dice)
            avg_iou = sum(all_iou) / len(all_iou)
            
            print(f"  Average Dice: {avg_dice:.4f}")
            print(f"  Average IoU: {avg_iou:.4f}")
        
        # Total slices processed
        total_slices = sum(r['num_slices'] for r in successful_results)
        print(f"  Total slices processed: {total_slices:,}")


if __name__ == '__main__':
    main()
