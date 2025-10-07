#!/usr/bin/env python3
"""
Batch processing example for the stim_transformations library.

This script demonstrates how to process multiple images in batch,
useful for processing large datasets of stimulus images.
"""

import sys
import os
import glob
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path to import stim_transformations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stim_transformations as stf


def create_sample_dataset(output_dir: str, num_images: int = 10):
    """Create a sample dataset of images for batch processing."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} sample images in {output_dir}...")
    
    for i in range(num_images):
        # Create random shapes at different positions and sizes
        img = np.zeros((200, 200), dtype=np.uint8)
        
        # Random shape type
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
        
        if shape_type == 'circle':
            center = (np.random.randint(50, 150), np.random.randint(50, 150))
            radius = np.random.randint(20, 40)
            cv2.circle(img, center, radius, 255, -1)
            
        elif shape_type == 'rectangle':
            x1 = np.random.randint(20, 100)
            y1 = np.random.randint(20, 100)
            x2 = x1 + np.random.randint(30, 80)
            y2 = y1 + np.random.randint(30, 80)
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
            
        else:  # triangle
            pts = np.array([
                [np.random.randint(50, 150), np.random.randint(50, 100)],
                [np.random.randint(50, 150), np.random.randint(100, 150)],
                [np.random.randint(50, 150), np.random.randint(50, 100)]
            ], np.int32)
            cv2.fillPoly(img, [pts], 255)
        
        # Save image
        filename = f'sample_{i:03d}.png'
        cv2.imwrite(os.path.join(output_dir, filename), img)
    
    print(f"Created {num_images} sample images")


def batch_center_images(input_dir: str, output_dir: str, reference_path: str = None):
    """Center all images in a directory to a reference image."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference image if provided
    img_ref = None
    if reference_path and os.path.exists(reference_path):
        img_ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        print(f"Using reference image: {reference_path}")
    else:
        print("No reference image provided, centering to image center")
    
    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpeg')))
    
    print(f"Processing {len(image_files)} images...")
    
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Center image
            centered = stf.setup(img, operation='center', img_ref=img_ref)
            
            # Save centered image
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f'centered_{filename}')
            cv2.imwrite(output_path, centered)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Centering completed in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(image_files):.3f} seconds")


def batch_scale_images(input_dir: str, output_dir: str, reference_path: str):
    """Scale all images in a directory to match reference dimensions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference image
    img_ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if img_ref is None:
        raise ValueError(f"Could not load reference image: {reference_path}")
    
    print(f"Using reference image: {reference_path}")
    print(f"Target dimensions: {img_ref.shape}")
    
    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpeg')))
    
    print(f"Processing {len(image_files)} images...")
    
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Scale image
            scaled = stf.setup(img, operation='scale', img_ref=img_ref)
            
            # Save scaled image
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f'scaled_{filename}')
            cv2.imwrite(output_path, scaled)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Scaling completed in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(image_files):.3f} seconds")


def batch_skeletonize_images(input_dir: str, output_dir: str):
    """Skeletonize all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpeg')))
    
    print(f"Processing {len(image_files)} images...")
    
    start_time = time.time()
    
    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Skeletonize image
            binary, skeleton = stf.setup(img, operation='skeleton')
            
            # Save skeleton image
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            binary_path = os.path.join(output_dir, f'binary_{filename}')
            skeleton_path = os.path.join(output_dir, f'skeleton_{filename}')
            
            cv2.imwrite(binary_path, binary)
            cv2.imwrite(skeleton_path, skeleton)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Skeletonization completed in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(image_files):.3f} seconds")


def create_summary_visualization(input_dir: str, output_dir: str, num_samples: int = 6):
    """Create a summary visualization of processed images."""
    # Get sample images
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpeg')))
    
    if len(image_files) == 0:
        print("No images found for visualization")
        return
    
    # Select random samples
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, len(sample_files), figsize=(3*len(sample_files), 6))
    if len(sample_files) == 1:
        axes = axes.reshape(2, 1)
    
    for i, img_path in enumerate(sample_files):
        # Load original image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load processed images if they exist
        filename = os.path.basename(img_path)
        centered_path = os.path.join(output_dir, f'centered_{filename}')
        scaled_path = os.path.join(output_dir, f'scaled_{filename}')
        skeleton_path = os.path.join(output_dir, f'skeleton_{filename}')
        
        # Original image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Processed image (show first available)
        if os.path.exists(centered_path):
            processed = cv2.imread(centered_path, cv2.IMREAD_GRAYSCALE)
            title = 'Centered'
        elif os.path.exists(scaled_path):
            processed = cv2.imread(scaled_path, cv2.IMREAD_GRAYSCALE)
            title = 'Scaled'
        elif os.path.exists(skeleton_path):
            processed = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
            title = 'Skeleton'
        else:
            processed = img
            title = 'No processed version'
        
        axes[1, i].imshow(processed, cmap='gray')
        axes[1, i].set_title(f'{title} {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_processing_summary.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Summary visualization saved to {output_dir}/batch_processing_summary.png")


def main():
    """Run batch processing examples."""
    print("Stimulus Transformations Library - Batch Processing Examples")
    print("=" * 60)
    
    # Create directories
    input_dir = "sample_dataset"
    output_dir = "processed_dataset"
    reference_path = "reference.png"
    
    try:
        # Create sample dataset
        create_sample_dataset(input_dir, num_images=20)
        
        # Create a reference image
        ref_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(ref_img, (100, 100), 60, 255, -1)
        cv2.imwrite(reference_path, ref_img)
        print(f"Created reference image: {reference_path}")
        
        print("\n" + "="*50)
        print("BATCH CENTERING")
        print("="*50)
        batch_center_images(input_dir, os.path.join(output_dir, "centered"), reference_path)
        
        print("\n" + "="*50)
        print("BATCH SCALING")
        print("="*50)
        batch_scale_images(input_dir, os.path.join(output_dir, "scaled"), reference_path)
        
        print("\n" + "="*50)
        print("BATCH SKELETONIZATION")
        print("="*50)
        batch_skeletonize_images(input_dir, os.path.join(output_dir, "skeletonized"))
        
        print("\n" + "="*50)
        print("CREATING SUMMARY VISUALIZATION")
        print("="*50)
        create_summary_visualization(input_dir, os.path.join(output_dir, "centered"))
        
        print("\n" + "="*50)
        print("BATCH PROCESSING COMPLETED!")
        print("="*50)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print("Check the output directories for processed images.")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        print("Make sure all dependencies are installed correctly.")


if __name__ == "__main__":
    main()
