import os
import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import trimesh
import numpy as np
import requests
from io import BytesIO
import zipfile

import trimesh
import numpy as np
import torch
from pyassimp import load as assimp_load, release as assimp_release

import trimesh
import numpy as np

# call it before or during batch
def compute_occupancy_labels(bounds, voxel_resolution, mesh_path):
    # Define bounds and a small voxel resolution for demonstration
    bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])

    # Create a uniform grid along each axis
    x = torch.linspace(bbox_min[0], bbox_max[0], steps=voxel_resolution + 1)
    y = torch.linspace(bbox_min[1], bbox_max[1], steps=voxel_resolution + 1)
    z = torch.linspace(bbox_min[2], bbox_max[2], steps=voxel_resolution + 1)

    # Generate the 3D grid
    xs, ys, zs = torch.meshgrid(x, y, z, indexing='ij')
    positions = torch.stack((xs, ys, zs), dim=-1)
    positions = positions.reshape(-1, 3)

    mesh = trimesh.load(mesh_path)
    # positions: numpy array of shape [N, 3]
    inside_mask = mesh.contains(positions)
    # Convert boolean mask to float labels: 1.0 for inside, 0.0 for outside
    labels = inside_mask.astype(np.float32)
    return labels


def load_mesh(file_path):
    """
    Load a 3D mesh from various file formats using trimesh and pyassimp.

    Parameters:
    - file_path (str): Path to the 3D mesh file.

    Returns:
    - mesh (trimesh.Trimesh): Loaded mesh object.
    """
    try:
        # Attempt to load using trimesh
        mesh = trimesh.load(file_path)
        if mesh.is_empty:
            raise ValueError("Loaded mesh is empty.")
    except Exception as e:
        print(f"trimesh failed to load {file_path}: {e}")
        try:
            # Attempt to load using pyassimp
            scene = assimp_load(file_path)
            if not scene.meshes:
                raise ValueError("No meshes found in the file.")
            # Convert the first mesh to a trimesh.Trimesh object
            mesh = trimesh.Trimesh(vertices=scene.meshes[0].vertices,
                                   faces=scene.meshes[0].faces)
            assimp_release(scene)
        except Exception as e:
            print(f"pyassimp failed to load {file_path}: {e}")
            raise ValueError(f"Failed to load mesh from {file_path}")
    return mesh


def load_and_preprocess_mesh(file_path, n_points=81920):
    """
    Load a 3D mesh from an OBJ or GLB file, sample points, and preprocess into a point cloud tensor.

    Parameters:
    - file_path (str): Path to the input 3D mesh file (.obj or .glb).
    - n_points (int): Number of points to sample from the mesh. Default is 81,920.

    Returns:
    - torch.Tensor: A tensor containing the sampled and normalized point cloud data with normals.
    """
    # Load the mesh from the file
    mesh = trimesh.load(file_path)

    # If the loaded file is a Scene (common for .glb), extract the first geometry
    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries into a single mesh
        mesh = trimesh.util.concatenate(mesh.geometry.values())

    # Sample points uniformly from the mesh surface and get face indices
    points, face_indices = mesh.sample(n_points, return_index=True)

    # Retrieve normals from the corresponding faces
    normals = mesh.face_normals[face_indices]

    point_cloud = normalize_points(points, normals)

    # Convert the numpy array to a PyTorch tensor
    point_cloud_tensor = torch.from_numpy(point_cloud).float()
    return point_cloud_tensor

def normalize_points(points, normals):
    # Normalize the point cloud: center it and scale to a unit sphere
    points_mean = points.mean(axis=0)
    points_centered = points - points_mean
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / scale

    # Concatenate the normalized points with their normals
    # Each point now has 6 channels: [x, y, z, nx, ny, nz]
    # point_cloud = np.concatenate([points_normalized, normals], axis=1)

    return points_normalized, normals


class ObjaverseXLDataset(Dataset):
    def __init__(self, hf_dataset, transform=None,
                 download_dir='objaverse_models', num_points=81920,
                 bounds=1, voxel_resolution=512
                 ):
        self.dataset = hf_dataset
        self.download_urls = hf_dataset['fileIdentifier']
        self.ids = hf_dataset['sha256']
        self.fileType = hf_dataset['fileType']
        self.transform = transform
        self.download_dir = download_dir
        self.num_points = num_points
        # This is the boundary in the 3D space, normally 1 due to normalization of data
        # need to be same as in vae
        self.bounds = bounds
        # This is the resolution to check for how many points to check
        # need to be same as in vae
        self.voxel_resolution = voxel_resolution
        os.makedirs(self.download_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def download_and_extract(self, url, obj_id, file_type, bounds=1):
        response = requests.get(url)
        if response.status_code == 200:
            print("Downloading", url)
            file_path = os.path.join(self.download_dir, f"{obj_id}.{file_type}")

            with open(file_path, 'wb') as f:
                f.write(response.content)

            return file_path
        else:
            raise Exception(f"Failed to download {url}")

    def __getitem__(self, idx):
        # sample = self.dataset[idx]
        # obj_id = sample['id']
        # obj_url = sample['url']
        obj_id = self.ids[idx]
        obj_url = self.download_urls[idx]
        file_type = self.fileType[idx]

        file_path = os.path.join(self.download_dir, f"{obj_id}.{file_type}")
        if not os.path.exists(file_path):
            self.download_and_extract(obj_url, obj_id, file_type)

        # mesh = load_objs_as_meshes([obj_path])
        mesh = load_mesh(file_path)
        points, normals = sample_points_from_meshes(mesh, num_samples=self.num_points, return_normals=True)

        # either precompute or run during training
        labels = compute_occupancy_labels(self.bounds, self.voxel_resolution, file_path)

        # normalize or not, need testing
        point_cloud, normals = normalize_points(points, normals)
        # print(point_cloud.shape)

        # return torch.from_numpy(point_cloud).float()
        return point_cloud, normals, labels

class DummyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, download_dir='objaverse_models', num_points=81920,
                 bounds=1, voxel_resolution=512):
        self.dataset = hf_dataset
        self.num_points = num_points
        self.bounds = bounds
        self.voxel_resolution = voxel_resolution

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # normalize or not, need testing
        point_cloud = np.random.randn(self.num_points, 3)
        normals = np.random.randn(self.num_points, 3)
        # print(point_cloud.shape)
        labels = np.random.randint(0, 1, ((self.voxel_resolution+1)**3, ))

        return torch.from_numpy(point_cloud).float(), torch.from_numpy(normals).float(), torch.from_numpy(labels).float()
