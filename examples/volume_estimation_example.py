"""
Example script demonstrating tumor volume estimation
"""
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.volume_estimation import create_volume_estimator


def main():
    """Example usage of tumor volume estimation"""
    
    print("=== Tumor Volume Estimation Example ===\n")
    
    # Create volume estimator
    estimator = create_volume_estimator()
    
    # Example 1: Create synthetic segmentation data
    print("1. Creating synthetic tumor segmentation...")
    
    # Create a synthetic 3D tumor (ellipsoid shape)
    shape = (128, 128, 64)  # (x, y, z)
    segmentation = np.zeros(shape, dtype=np.int32)
    
    # Create tumor center
    center = [shape[0]//2, shape[1]//2, shape[2]//2]
    
    # Create ellipsoid tumor
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Ellipsoid equation
                dx = (x - center[0]) / 20.0
                dy = (y - center[1]) / 15.0
                dz = (z - center[2]) / 10.0
                
                if dx**2 + dy**2 + dz**2 <= 1:
                    # Create different tumor regions
                    if dx**2 + dy**2 + dz**2 <= 0.3:
                        segmentation[x, y, z] = 1  # Necrotic core
                    elif dx**2 + dy**2 + dz**2 <= 0.7:
                        segmentation[x, y, z] = 3  # Enhancing tumor
                    else:
                        segmentation[x, y, z] = 2  # Edema
    
    print(f"   Created synthetic tumor with shape: {shape}")
    print(f"   Total tumor voxels: {np.sum(segmentation > 0):,}")
    
    # Example 2: Calculate volume with different voxel spacings
    print("\n2. Calculating tumor volume with different voxel spacings...")
    
    voxel_spacings = [
        (1.0, 1.0, 1.0),    # 1mm isotropic
        (0.5, 0.5, 3.0),    # Typical clinical spacing
        (0.8, 0.8, 5.0),    # Another clinical example
    ]
    
    for spacing in voxel_spacings:
        volume = estimator.calculate_tumor_volume(segmentation, spacing)
        print(f"   Spacing {spacing}: {volume:.2f} mm³")
    
    # Example 3: Calculate depth measurements
    print("\n3. Calculating tumor depth measurements...")
    
    spacing = (1.0, 1.0, 1.0)
    depth = estimator.calculate_tumor_depth(segmentation, spacing)
    
    print(f"   Anterior-Posterior depth: {depth['depth_ap_mm']:.2f} mm")
    print(f"   Superior-Inferior depth: {depth['depth_si_mm']:.2f} mm")
    print(f"   Left-Right depth: {depth['depth_lr_mm']:.2f} mm")
    print(f"   Maximum depth: {depth['max_depth_mm']:.2f} mm")
    
    # Example 4: Count tumor slices
    print("\n4. Counting tumor slices...")
    
    axial_slices = estimator.count_tumor_slices(segmentation, axis=2)
    coronal_slices = estimator.count_tumor_slices(segmentation, axis=1)
    sagittal_slices = estimator.count_tumor_slices(segmentation, axis=0)
    
    print(f"   Axial slices with tumor: {axial_slices}")
    print(f"   Coronal slices with tumor: {coronal_slices}")
    print(f"   Sagittal slices with tumor: {sagittal_slices}")
    
    # Example 5: Analyze tumor composition
    print("\n5. Analyzing tumor composition...")
    
    composition = estimator.analyze_tumor_composition(segmentation, spacing)
    
    print("   Tumor composition:")
    for comp_name, comp_data in composition.items():
        print(f"     {comp_name.capitalize()}:")
        print(f"       Volume: {comp_data['volume_mm3']:.2f} mm³")
        print(f"       Percentage: {comp_data['percentage']:.1f}%")
        print(f"       Slices: {comp_data['tumor_slices']}")
    
    # Example 6: Calculate surface area
    print("\n6. Estimating tumor surface area...")
    
    surface_area = estimator.calculate_surface_area(segmentation, spacing)
    print(f"   Surface area: {surface_area:.2f} mm²")
    
    # Example 7: Calculate derived metrics
    print("\n7. Calculating derived metrics...")
    
    total_volume = composition['necrotic']['volume_mm3'] + composition['edema']['volume_mm3'] + composition['enhancing']['volume_mm3']
    
    if total_volume > 0:
        volume_surface_ratio = total_volume / surface_area
        equivalent_radius = (3 * total_volume / (4 * np.pi)) ** (1/3)
        
        print(f"   Volume/Surface ratio: {volume_surface_ratio:.3f} mm")
        print(f"   Equivalent sphere radius: {equivalent_radius:.2f} mm")
    
    # Example 8: Create comprehensive analysis
    print("\n8. Creating comprehensive analysis...")
    
    analysis = {
        'tumor_volume_mm3': total_volume,
        'tumor_slices': axial_slices,
        'tumor_depth_mm': depth,
        'tumor_surface_area_mm2': surface_area,
        'tumor_composition': composition,
        'voxel_spacing_mm': spacing
    }
    
    print("   Comprehensive analysis:")
    print(f"     Total volume: {analysis['tumor_volume_mm3']:.2f} mm³")
    print(f"     Tumor slices: {analysis['tumor_slices']}")
    print(f"     Max depth: {analysis['tumor_depth_mm']['max_depth_mm']:.2f} mm")
    print(f"     Surface area: {analysis['tumor_surface_area_mm2']:.2f} mm²")
    
    print("\n=== Example completed ===")
    
    # Example 9: Real-world usage (if segmentation files exist)
    print("\n9. Real-world usage example...")
    
    # Look for existing segmentation files
    segmentation_files = []
    search_paths = [
        "segmentation_results/*prediction*.nii",
        "segmentation_with_volume/*segmentation*.nii",
        "checkpoints/unet/*prediction*.nii"
    ]
    
    import glob
    for pattern in search_paths:
        files = glob.glob(pattern)
        segmentation_files.extend(files)
    
    if segmentation_files:
        print(f"   Found {len(segmentation_files)} segmentation files")
        
        # Analyze first file as example
        seg_file = segmentation_files[0]
        print(f"   Analyzing: {os.path.basename(seg_file)}")
        
        try:
            result = estimator.estimate_from_nifti(seg_file)
            
            print(f"     Volume: {result['tumor_volume_mm3']:.2f} mm³")
            print(f"     Slices: {result['tumor_slices_axial']}")
            print(f"     Max depth: {result['tumor_depth_mm']['max_depth_mm']:.2f} mm")
            
            # Print composition
            if 'tumor_composition' in result:
                print("     Composition:")
                for comp_name, comp_data in result['tumor_composition'].items():
                    if comp_data['volume_mm3'] > 0:
                        print(f"       {comp_name}: {comp_data['volume_mm3']:.2f} mm³ ({comp_data['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"     Error: {str(e)}")
    else:
        print("   No segmentation files found for real-world example")
        print("   To test with real data:")
        print("   1. Train a segmentation model: python scripts/train_segmentation.py")
        print("   2. Run inference: python scripts/segmentation_with_volume.py")
        print("   3. Then use the volume estimation tools")


if __name__ == '__main__':
    main()
