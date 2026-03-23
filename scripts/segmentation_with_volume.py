"""
Script for tumor segmentation with automatic volume estimation
"""
import os
import sys
import argparse
import json
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.segmentation_inference_with_volume import create_segmentation_inference_with_volume


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tumor segmentation with volume estimation')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (patient folder or directory with patient folders)')
    parser.add_argument('--output_dir', type=str, default='segmentation_with_volume',
                       help='Output directory for results')
    parser.add_argument('--slice_axis', type=int, default=2,
                       choices=[0, 1, 2],
                       help='Axis for slice extraction (0=sagittal, 1=coronal, 2=axial)')
    parser.add_argument('--save_probabilities', action='store_true',
                       help='Save probability maps')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save slice visualizations')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate HTML volume report')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories for patient folders')
    parser.add_argument('--print_results', action='store_true',
                       help='Print results to console')
    
    return parser.parse_args()


def find_patient_folders(input_path: str, recursive: bool = False) -> list:
    """Find patient folders in input path"""
    patient_folders = []
    
    if os.path.isdir(input_path):
        if recursive:
            # Search recursively for BraTS folders
            search_patterns = [
                os.path.join(input_path, "**/BraTS20_*"),
                os.path.join(input_path, "**/BraTS21_*"),
                os.path.join(input_path, "**/*BraTS*")
            ]
            
            for pattern in search_patterns:
                folders = glob.glob(pattern, recursive=True)
                for folder in folders:
                    if os.path.isdir(folder):
                        patient_folders.append(folder)
        else:
            # Search only in immediate directory
            for item in os.listdir(input_path):
                item_path = os.path.join(input_path, item)
                if os.path.isdir(item_path) and 'BraTS' in item:
                    patient_folders.append(item_path)
    
    # Remove duplicates and sort
    patient_folders = sorted(list(set(patient_folders)))
    
    return patient_folders


def main():
    """Main function"""
    args = parse_args()
    
    print("Tumor Segmentation with Volume Estimation")
    print("=" * 50)
    
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
    
    # Create enhanced inference instance
    print("Loading model...")
    inference = create_segmentation_inference_with_volume(
        checkpoint_path=args.model_path,
        device=device
    )
    
    # Find patient folders
    patient_folders = find_patient_folders(args.input, args.recursive)
    
    if not patient_folders:
        print("No patient folders found. Please check your input path.")
        return
    
    print(f"Found {len(patient_folders)} patient folders")
    
    # Process patients
    print(f"\nProcessing {len(patient_folders)} patients...")
    
    results = inference.batch_predict_with_volume(
        patient_folders=patient_folders,
        slice_axis=args.slice_axis,
        return_probabilities=args.save_probabilities,
        output_dir=args.output_dir,
        save_summary=True
    )
    
    # Generate visualizations if requested
    if args.save_visualizations:
        print("\nGenerating visualizations...")
        
        # Use base inference for visualizations
        for i, patient_folder in enumerate(patient_folders):
            try:
                patient_id = os.path.basename(patient_folder)
                print(f"  Visualizing {i+1}/{len(patient_folders)}: {patient_id}")
                
                # Generate visualizations using base inference
                vis_result = inference.base_inference.predict_patient_volume(
                    patient_folder=patient_folder,
                    slice_axis=args.slice_axis,
                    return_probabilities=False
                )
                
                # Save visualizations
                vis_dir = os.path.join(args.output_dir, "visualizations", patient_id)
                os.makedirs(vis_dir, exist_ok=True)
                
                # Generate sample slice visualizations
                import numpy as np
                from src.utils.helpers import set_seed
                set_seed(42)
                
                # Load modality volumes
                modality_volumes = {}
                for modality in inference.base_inference.config['modalities']:
                    modality_file = glob.glob(os.path.join(patient_folder, f"*{modality}*.nii*"))[0]
                    import nibabel as nib
                    img = nib.load(modality_file)
                    modality_volumes[modality] = img.get_fdata()
                
                # Select sample slices
                num_slices = modality_volumes[inference.base_inference.config['modalities'][0]].shape[args.slice_axis]
                sample_indices = [num_slices // 4, num_slices // 2, 3 * num_slices // 4]
                
                for j, slice_idx in enumerate(sample_indices):
                    if 0 <= slice_idx < num_slices:
                        # Extract modality slices
                        modality_slices = []
                        for modality in inference.base_inference.config['modalities']:
                            volume = modality_volumes[modality]
                            
                            if args.slice_axis == 0:
                                slice_data = volume[slice_idx, :, :]
                            elif args.slice_axis == 1:
                                slice_data = volume[:, slice_idx, :]
                            else:
                                slice_data = volume[:, :, slice_idx]
                            
                            modality_slices.append(slice_data)
                        
                        # Get prediction slice
                        prediction_volume = vis_result['prediction_volume']
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
                            seg_data[seg_data == 4] = 3
                            
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
                        inference.base_inference.visualize_slice_prediction(
                            modality_slices=modality_slices,
                            prediction=pred_slice,
                            ground_truth=ground_truth_slice,
                            slice_idx=slice_idx,
                            save_path=vis_path
                        )
                
                print(f"    Visualizations saved to {vis_dir}")
                
            except Exception as e:
                print(f"    Error visualizing {patient_id}: {str(e)}")
    
    # Generate HTML report if requested
    if args.generate_report:
        print("\nGenerating HTML report...")
        report_path = os.path.join(args.output_dir, "volume_report.html")
        inference.generate_volume_report(results, report_path)
    
    # Print results if requested
    if args.print_results:
        print("\nResults Summary:")
        print("-" * 80)
        
        for result in results:
            if 'error' in result:
                print(f"❌ {result['patient_id']}: {result['error']}")
            else:
                print(f"✅ {result['patient_id']}:")
                print(f"   Volume: {result['tumor_volume_mm3']:.2f} mm³")
                print(f"   Slices: {result['tumor_slices_axial']} (axial)")
                print(f"   Max depth: {result['tumor_depth_mm']['max_depth_mm']:.2f} mm")
                print(f"   Surface area: {result['tumor_surface_area_mm2']:.2f} mm²")
                
                # Print composition
                if 'tumor_composition' in result:
                    print("   Composition:")
                    for comp_name, comp_data in result['tumor_composition'].items():
                        print(f"     {comp_name}: {comp_data['volume_mm3']:.2f} mm³ ({comp_data['percentage']:.1f}%)")
                
                print()
    
    print(f"\nSegmentation with volume estimation completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
