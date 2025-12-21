import os
import glob
import h5py
import numpy as np
import SimpleITK as sitk

def save_h5(image_path, label_path, scribble_path, save_path):
    # Load Image
    img_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    
    # Load Label
    lab_itk = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(lab_itk).astype(np.uint8)
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        
        # Save scribble if path exists (only for Training set)
        if scribble_path and os.path.exists(scribble_path):
            scr_itk = sitk.ReadImage(scribble_path)
            scribble = sitk.GetArrayFromImage(scr_itk).astype(np.uint8)
            f.create_dataset('scribble', data=scribble, compression="gzip")

def process_dataset(root_path, mode="Tr"):
    img_dir = os.path.join(root_path, f"images{mode}")
    lab_dir = os.path.join(root_path, f"labels{mode}")
    scr_dir = os.path.join(root_path, f"scribbles{mode}") if mode == "Tr" else None
    
    output_dir = os.path.join(root_path, f"h5_{mode}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    print(f"Processing {mode} set: {len(img_list)} volumes.")
    
    for img_path in img_list:
        file_name = os.path.basename(img_path)
        lab_path = os.path.join(lab_dir, file_name)
        scr_path = os.path.join(scr_dir, file_name) if scr_dir else None
        
        h5_save_path = os.path.join(output_dir, file_name.replace('.nii.gz', '.h5'))
        
        try:
            save_h5(img_path, lab_path, scr_path, h5_save_path)
            print(f"Done: {file_name}")
        except Exception as e:
            print(f"Failed: {file_name} | Error: {e}")

if __name__ == "__main__":
    # Define your root directory
    base_path = "/home/data/sxl/PS-Seg/data/Word/WORD-V0.1.0-Admin_cropWL"
    
    # Process Training set (image, label, scribble)
    process_dataset(base_path, mode="Tr", output_dir = "/home/data/sxl/PS-Seg/data/Word/Word_3d/Abdomen_Tr_volumes")
    
    # Process Validation set (image, label)
    process_dataset(base_path, mode="Val", output_dir = "/home/data/sxl/PS-Seg/data/Word/Word_3d/Abdomen_Val_volumes")