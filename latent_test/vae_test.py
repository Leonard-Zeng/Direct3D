import argparse
import copy
import numpy as np
import pandas as pd
import torch
import trimesh
from direct3d.pipeline import Direct3dPipeline
import pandas

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a 3D mesh and save its latent representation.")
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
        help="Device to perform computations on: 'cpu', 'cuda', or 'mps'. Default is 'cpu'."
    )
    parser.add_argument(
        '--input_mesh', type=str, required=True,
        help="Path to the input 3D mesh file (e.g., 'output.obj')."
    )
    parser.add_argument(
        '--output_latent', type=str, default='latent.npy',
        help="Filename for the output latent NumPy file. Default is 'latent.npy'."
    )
    parser.add_argument(
        '--n_points', type=int, default=81920,
        help="Number of points to sample from the mesh. Default is 81920."
    )
    return parser.parse_args()

def load_and_preprocess_mesh(obj_filename, n_points=81920):
    # Load the mesh from the OBJ file
    mesh = trimesh.load(obj_filename)

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

    # Preprocess the point cloud
    # x = preprocess_point_cloud(point_cloud_tensor, vae.fourier_embedder)

    # Encode the processed input to obtain the latent representation
    vae.eval()
    with torch.no_grad():
        latent, posterior = vae.encode(pc.unsqueeze(0), feats.unsqueeze(0))

    # return posterior.mean
    return latent

def main():
    args = parse_arguments()

    assert (args.input_mesh is not None) or (args.item is not None)

    # Determine the computation device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Load the Direct3D pipeline and VAE
    pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
    vae = copy.deepcopy(pipeline.vae.eval()).to(torch.float32).to(device)
    del pipeline  # Free up memory

    latent = generate_latent(args.input_mesh, args.n_points, vae, device)
    del vae
    # Save the latent representation to a NumPy file
    # np.save(args.output_latent, latent_numpy)
    print(f"Latent representation generated with shape {latent.shape}")

    # initialize it again:
    # print(latent.device)
    pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
    pipeline.to(device)
    # print(latent.device)
    mesh = pipeline.vae_reconstruction(latent)["meshes"][0]
    mesh.export("recovered_output.obj")
    print("saved recovered_output.obj")

if __name__ == "__main__":
    main()
