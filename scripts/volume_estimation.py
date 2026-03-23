"""
Script for tumor volume estimation from segmentation masks
"""
import os
import sys
import argparse
import json
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.volume_estimation import create_volume_estimator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Estimate tumor volume from segmentation masks')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input path (segmentation file or directory with segmentations)')
    parser.add_argument('--reference_dir', type=str, default=None,
                       help='Directory with reference NIfTI files for voxel spacing')
    parser.add_argument('--output', type=str, default='volume_analysis.json',
                       help='Output file for results')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories for segmentation files')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary statistics')
    parser.add_argument('--print_results', action='store_true',
                       help='Print results to console')
    parser.add_argument('--file_pattern', type=str, default='*prediction*.nii',
                       help='Pattern to match segmentation files')
    
    return parser.parse_args()


def find_reference_file(seg_path: str, reference_dir: str) -> str:
    """Find corresponding reference file for segmentation"""
    # Extract patient ID from segmentation path
    patient_id = os.path.basename(seg_path).replace('_prediction.nii', '').replace('.nii', '')
    
    # Look for reference files with patient ID
    reference_patterns = [
        f"*{patient_id}*t1*.nii*",
        f"*{patient_id}*flair*.nii*",
        f"*{patient_id}*.nii*"
    ]
    
    for pattern in reference_patterns:
        matches = glob.glob(os.path.join(reference_dir, pattern))
        if matches:
            return matches[0]
    
    return None


def main():
    """Main volume estimation function"""
    args = parse_args()
    
    print("Tumor Volume Estimation")
    print("=" * 50)
    
    # Create volume estimator
    estimator = create_volume_estimator()
    
    # Find segmentation files
    segmentation_files = []
    
    if os.path.isfile(args.input):
        segmentation_files = [args.input]
    elif os.path.isdir(args.input):
        if args.recursive:
            segmentation_files = glob.glob(os.path.join(args.input, "**", args.file_pattern), recursive=True)
        else:
            segmentation_files = glob.glob(os.path.join(args.input, args.file_pattern))
    
    segmentation_files = sorted(segmentation_files)
    
    if not segmentation_files:
        print(f"No segmentation files found matching pattern: {args.file_pattern}")
        return
    
    print(f"Found {len(segmentation_files)} segmentation files")
    
    # Find reference files if reference directory is provided
    reference_files = [None] * len(segmentation_files)
    
    if args.reference_dir and os.path.exists(args.reference_dir):
        print(f"Looking for reference files in: {args.reference_dir}")
        
        for i, seg_file in enumerate(segmentation_files):
            ref_file = find_reference_file(seg_file, args.reference_dir)
            reference_files[i] = ref_file
            
            if ref_file:
                print(f"  {os.path.basename(seg_file)} -> {os.path.basename(ref_file)}")
            else:
                print(f"  {os.path.basename(seg_file)} -> No reference found")
    
    # Analyze segmentations
    print(f"\nAnalyzing {len(segmentation_files)} segmentations...")
    
    results = estimator.batch_analyze(
        segmentation_paths=segmentation_files,
        reference_paths=reference_files,
        output_path=args.output
    )
    
    # Print results if requested
    if args.print_results:
        print("\nResults:")
        print("-" * 80)
        
        for result in results:
            if 'error' in result:
                print(f"❌ {result['patient_id']}: {result['error']}")
            else:
                print(f"✅ {result['patient_id']}:")
                print(f"   Volume: {result['tumor_volume_mm3']:.2f} mm³")
                print(f"   Slices (axial): {result['tumor_slices_axial']}")
                print(f"   Max depth: {result['tumor_depth_mm']['max_depth_mm']:.2f} mm")
                print(f"   Surface area: {result['tumor_surface_area_mm2']:.2f} mm²")
                
                # Print composition
                print("   Composition:")
                for comp_name, comp_data in result['tumor_composition'].items():
                    print(f"     {comp_name}: {comp_data['volume_mm3']:.2f} mm³ ({comp_data['percentage']:.1f}%)")
                
                print()
    
    # Generate summary if requested
    if args.summary:
        print("\nSummary Statistics:")
        print("-" * 50)
        
        summary = estimator.generate_summary_report(results)
        
        print(f"Total patients: {summary['total_patients']}")
        print(f"Successful analyses: {summary['successful_analyses']}")
        print(f"Failed analyses: {summary['failed_analyses']}")
        
        if 'volume_stats' in summary:
            vs = summary['volume_stats']
            print(f"\nVolume Statistics (mm³):")
            print(f"  Mean: {vs['mean_mm3']:.2f} ± {vs['std_mm3']:.2f}")
            print(f"  Range: {vs['min_mm3']:.2f} - {vs['max_mm3']:.2f}")
            print(f"  Median: {vs['median_mm3']:.2f}")
        
        if 'slice_stats' in summary:
            ss = summary['slice_stats']
            print(f"\nSlice Statistics:")
            print(f"  Mean: {ss['mean_slices']:.1f} ± {ss['std_slices']:.1f}")
            print(f"  Range: {ss['min_slices']} - {ss['max_slices']}")
            print(f"  Median: {ss['median_slices']}")
        
        if 'composition_stats' in summary:
            print(f"\nComposition Statistics (%):")
            for class_name, stats in summary['composition_stats'].items():
                print(f"  {class_name}: {stats['mean_percentage']:.1f} ± {stats['std_percentage']:.1f}")
        
        # Save summary
        summary_path = args.output.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nVolume estimation completed!")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
