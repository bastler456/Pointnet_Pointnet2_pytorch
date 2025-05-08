import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import os
import argparse
import numpy as np
from data_utils.pcdDataLoader import HierarchicalPointCloudDataset
import open3d as o3d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['scanned', "bottle", 'box']
numb_of_classes: int = len(classes)
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def main():
    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    NUM_CLASSES = numb_of_classes


    pcd = o3d.io.read_point_cloud("data/scene.pcd")
    o3d.visualization.draw_geometries([pcd])
    points_np = np.asarray(pcd.points)

    points = torch.tensor(points_np, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, N, 3)

    '''MODEL LOADING'''
    model_name = 'pointnet_sem_seg'
    MODEL = importlib.import_module(f'models.{model_name}')
    model = MODEL.get_model(NUM_CLASSES).to(device)
    checkpoint = torch.load("best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    CLASS_COLORS = np.array([
                    [255, 0, 0],    # Red
                    [0, 255, 0],    # Green
                    [0, 0, 255],    # Blue
                    [255, 255, 0],  # Yellow
                    [255, 0, 255],  # Magenta
                    [0, 255, 255],  # Cyan
                ], dtype=np.uint8)
    
    # Chunking parameters
    points_in_scene = points.shape
    max_batch_size = 100000
    num_full_chunks = int(points_in_scene[1] / max_batch_size)
    remainder = points_in_scene[1] % max_batch_size
    total_chunks = num_full_chunks + (1 if remainder > 0 else 0)
    chunk_size = int(points_in_scene[1] / total_chunks)  # number of points per chunk
    outputs = []

    # Process in chunks
    for i in range(0, points_np.shape[0], chunk_size):
        print(f"run {i}")
        chunk = points_np[i:i+chunk_size]  # shape: (chunk_size, 3)
        
        if chunk.shape[0] == 0:
            continue  # skip empty chunks at the end

        # Convert to tensor and permute to (1, 3, chunk_size)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

        # Run through model
        with torch.no_grad():  # avoid storing gradients if you're just inferring
            output = model(chunk_tensor)[0]
            print(f"Output tensor shape: {output.shape}")
        
        outputs.append(output.cpu())  # move to CPU and store

    # Concatenate all outputs
    final_output = torch.cat(outputs[:-1], dim=1) 

    print(f"Final output shape: {final_output.shape}")

    # preds = final_output.permute(0, 2, 1)      # (1, N, C)
    pred_labels = torch.argmax(final_output, dim=-1).squeeze(0).cpu().numpy()  # (N,)

    # Trim points to match dropped predictions
    points_np_trimmed = points_np[:len(pred_labels)]  # shape: (N, 3)

    # Map labels to colors
    colors = CLASS_COLORS[pred_labels % len(CLASS_COLORS)] / 255.0

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np_trimmed)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

   



if __name__ == '__main__':
    main()