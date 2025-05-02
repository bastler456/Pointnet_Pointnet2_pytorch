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

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=50000, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default="", help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % "log/sem_seg")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = numb_of_classes
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    pcd = o3d.io.read_point_cloud("data/scene.pcd")
    points_np = np.asarray(pcd.points)

    points = torch.tensor(points_np, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, N, 3)

    '''MODEL LOADING'''
    model_name = 'pointnet_sem_seg'
    MODEL = importlib.import_module(f'models.{model_name}')
    model = MODEL.get_model(NUM_CLASSES).cuda()
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
    points_in_scene = points_np.shape[0]  # total number of points
    max_batch_size = 100000

    # Calculate number of full chunks
    num_full_chunks = points_in_scene // max_batch_size
    remainder = points_in_scene % max_batch_size

    # Total number of chunks
    total_chunks = num_full_chunks + (1 if remainder > 0 else 0)

    
    chunk_size = points_in_scene // total_chunks  # number of points per chunk
    outputs = []

    # Process in chunks
    for i in range(0, points_np.shape[0], chunk_size):
        chunk = points_np[i:i+chunk_size]  # shape: (chunk_size, 3)
        
        if chunk.shape[0] == 0:
            continue  # skip empty chunks at the end

        # Convert to tensor and permute to (1, 3, chunk_size)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

        # Run through model
        with torch.no_grad():  # avoid storing gradients if you're just inferring
            output = model(chunk_tensor)  # shape: (1, 64, chunk_size)
        
        outputs.append(output[0].cpu())  # move to CPU and store

    # Concatenate all outputs
    final_output = torch.cat(outputs[:-1], dim=2)  # shape: (1, 64, total_points)

    print(f"Final output shape: {final_output.shape}")

    preds = final_output.permute(0, 2, 1)      # (1, N, C)
    pred_labels = torch.argmax(preds, dim=-1).squeeze(0).cpu().numpy()  # (N,)

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
    args = parse_args()
    main(args)