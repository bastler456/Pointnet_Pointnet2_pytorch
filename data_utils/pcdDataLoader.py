import os
import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np


class HierarchicalPointCloudDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory with class folders.
            split (str): 'train' or 'test'
            transform (callable, optional): Transform to apply to pointclouds.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_paths = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name, split)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.endswith('.pcd'):
                    pcd_path = os.path.join(class_path, fname)
                    label_path = pcd_path.replace('.pcd', '_label.npy')
                    self.data_paths.append((pcd_path, label_path))

        self.data_paths.sort()

    def __len__(self):
        return len(self.data_paths)

    def __preproc__(self, pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        num_points=10000
        N, D = points.shape
        if N >= num_points:
            idx = np.random.choice(N, num_points, replace=False)
            return points[idx]
        else:
            raise IOError("Pointcloud to small")
            pad = np.zeros((num_points - N, D), dtype=points.dtype)
            return np.concatenate([points, pad], axis=0)


    def __getitem__(self, idx):
        pcd_path, label_path = self.data_paths[idx]

        points = self.__preproc__(pcd_path)

        labels = np.load(label_path).astype(np.int64)

        points_tensor = torch.tensor(points, dtype=torch.float64)
        labels = labels[:points_tensor.shape[0]]

        if self.transform:
            points_tensor = self.transform(points_tensor)

        return points_tensor, torch.from_numpy(labels)

    def custom_collate_fn(self, batch):
        points_batch = [item[0] for item in batch]
        labels_batch = [item[1] for item in batch]  

        return points_batch, labels_batch 