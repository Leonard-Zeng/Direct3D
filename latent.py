import argparse
import copy
import numpy as np
import pandas as pd
import torch
import trimesh
from direct3d.pipeline import Direct3dPipeline


import numpy as np
import torch
import trimesh

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

    # Normalize the point cloud: center it and scale to a unit sphere
    points_mean = points.mean(axis=0)
    points_centered = points - points_mean
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / scale

    # Concatenate the normalized points with their normals
    # Each point now has 6 channels: [x, y, z, nx, ny, nz]
    point_cloud = np.concatenate([points_normalized, normals], axis=1)

    # Convert the numpy array to a PyTorch tensor
    point_cloud_tensor = torch.from_numpy(point_cloud).float()
    return point_cloud_tensor


def generate_latent(input_mesh, n_points, vae, device):
    # Load and preprocess the mesh
    point_cloud_tensor = load_and_preprocess_mesh(input_mesh, n_points=n_points).to(device).to(torch.float16)
    pc = point_cloud_tensor[:, :3]  # Spatial coordinates (N, 3)
    feats = point_cloud_tensor[:, 3:]  # Additional features (N, 3)

    # Encode the processed input to obtain the latent representation
    vae.eval()
    with torch.no_grad():
        latent, posterior = vae.encode(pc.unsqueeze(0), feats.unsqueeze(0))

    return latent, posterior

