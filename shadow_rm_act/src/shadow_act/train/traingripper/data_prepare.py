import h5py
import pandas as pd
import cv2
import numpy as np
import os
import glob

def process_actions_csv(original_csv_path, output_dir, base_name):
    """
    Process the original action CSV file and split it into two files for the left and right arms.

    Args:
        original_csv_path (str): Path to the original action CSV file.
        output_dir (str): Output directory.
        base_name (str): Base name of the HDF5 file (e.g., 'episode_16').
    """
    try:
        main_df = pd.read_csv(original_csv_path)
        
        # # Remove the first row of data (Optional, commented out)
        # if not main_df.empty:
        #     main_df = main_df.iloc[1:].copy()

        # --- Process Left Arm Data ---
        if main_df.shape[1] >= 7:
            left_df = main_df.iloc[:, 0:7]
            # Assuming the intention is 7 headers corresponding to 7 columns of data
            left_headers = [0, 1, 2, 3, 4, 5, 'left_gripper']
            left_df.columns = left_headers
            left_csv_path = os.path.join(output_dir, f'{base_name}_actions_left.csv')
            left_df.to_csv(left_csv_path, index=False)
            print(f"Successfully created left arm action file: {left_csv_path}")

        # --- Process Right Arm Data ---
        if main_df.shape[1] >= 14:
            right_df = main_df.iloc[:, 7:14]
            # Assuming the intention is 7 headers corresponding to 7 columns of data
            right_headers = [0, 1, 2, 3, 4, 5, 'right_gripper']
            right_df.columns = right_headers
            right_csv_path = os.path.join(output_dir, f'{base_name}_actions_right.csv')
            right_df.to_csv(right_csv_path, index=False)
            print(f"Successfully created right arm action file: {right_csv_path}")

    except Exception as e:
        print(f"Error occurred while processing action CSV: {e}")


def extract_data_from_hdf5(hdf5_path):
    """
    Extract action data to CSV and image data to video files from an HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    
    # Create a new output directory
    # output_dir = os.path.join(script_dir, 'data_prepare_result', base_name)
    output_dir = os.path.join(script_dir, data_folder_save, base_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"All results will be saved in: {output_dir}")

    try:
        with h5py.File(hdf5_path, 'r') as f:
            print("HDF5 file opened successfully, reading data...")
            
            # --- Extract Action Data to CSV ---
            action_dset = f.get('action')
            if isinstance(action_dset, h5py.Dataset):
                actions = action_dset[:]
                action_df = pd.DataFrame(actions)
                
                csv_path = os.path.join(output_dir, f'{base_name}_actions.csv')
                action_df.to_csv(csv_path, index=False)
                print(f"Successfully extracted action data to {csv_path}")

                # Process the generated actions.csv
                process_actions_csv(csv_path, output_dir, base_name)
            else:
                print("Warning: 'action' dataset not found in HDF5 file.")

            # --- Extract Image Data to Video ---
            obs_group = f.get('observations')
            if isinstance(obs_group, h5py.Group):
                image_group = obs_group.get('images')
                if isinstance(image_group, h5py.Group):
                    camera_views = ['cam_right', 'cam_left']
                    
                    for cam_name in camera_views:
                        image_dset = image_group.get(cam_name)
                        if isinstance(image_dset, h5py.Dataset):
                            images = image_dset[:]
                            
                            if not (isinstance(images, np.ndarray) and images.ndim == 4 and images.shape[0] > 0):
                                print(f"Warning: Image data format for '{cam_name}' is incorrect or empty.")
                                continue

                            num_frames, height, width, channels = images.shape
                            
                            # Set video properties
                            video_path = os.path.join(output_dir, f'{base_name}_{cam_name}.mp4')
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
                            fps = 30.0  # Default FPS, modify if exact FPS is known
                            
                            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                            print(f"Processing video for '{cam_name}'...")
                            for i in range(num_frames):
                                frame = images[i]
                                # OpenCV requires BGR format, convert if original is RGB
                                if channels == 3:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                out.write(frame)
                            
                            out.release()
                            print(f"Successfully created video: {video_path}")
                        else:
                            print(f"Warning: '{cam_name}' dataset not found in 'observations/images'.")
                else:
                    print("Warning: 'observations/images' group does not exist or is not a group.")
            else:
                print("Warning: 'observations' group not found in HDF5 file.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # # Assume script and HDF5 file are in the same directory (Commented out)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # hdf5_file_name = 'episode  _16.hdf5'
    # hdf5_file_path = os.path.join(script_dir, hdf5_file_name)

    # if os.path.exists(hdf5_file_path):
    #     extract_data_from_hdf5(hdf5_file_path)
    # else:
    #     print(f"Error: HDF5 file not found at '{hdf5_file_path}'")
    #     print("Please ensure the script and HDF5 file are in the same directory.")
    
    # Get the directory where the script is located
    
    data_folder_save = 'prepare_result_Tools'
    data_source_dir = '/home/rm/collect_data/Tools'
    
    # Match all episode_*.hdf5 files
    hdf5_files = glob.glob(os.path.join(data_source_dir, 'episode_*.hdf5'))

    if not hdf5_files:
        print(f"No episode_*.hdf5 files found in directory {data_source_dir}, please check.")
    else:
        for hdf5_file_path in hdf5_files:
            print(f"\nProcessing: {hdf5_file_path}")
            extract_data_from_hdf5(hdf5_file_path)
