import copy
import itertools
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from direct3d.pipeline import Direct3dPipeline
from direct3d.models import vae
from datasets import load_dataset
from ObjaverseDataset import ObjaverseXLDataset, DummyDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_ratio",
        type=float,
        default=0.01,
        help="Ratio for test split when downloading objaverse dataset."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.001,
        help="Ratio for test split in main dataset."
    )
    parser.add_argument(
        "--test_size_split",
        type=float,
        default=0.1,
        help="Second test split ratio for train/test splitting."
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.2,
        help="Validation split ratio for train/val splitting."
    )
    parser.add_argument(
        "--voxel_resolution",
        type=int,
        default=128,
        help="Voxel resolution used in DummyDataset and get_mesh_logits."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DataLoaders."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs."
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scale factor for KL regularization term."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Select device: 'cpu' or 'cuda'."
    )
    return parser.parse_args()

def load_pretrained_vae() -> vae.D3D_VAE:
    pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
    model_vae = copy.deepcopy(pipeline.vae)
    del pipeline # to release VRAM
    return model_vae

def download_objaverse(ratio=0.01):
    dataset = load_dataset("allenai/objaverse-xl")
    dataset = dataset['train'].train_test_split(test_size=ratio)['test']
    for data in dataset:
        print(data)
        break

def kl_regularization(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train_one_epoch(model, train_loader, val_loader, optimizer, bce, kl, kl_scale, device, voxel_resolution):
    for data in train_loader:
        optimizer.zero_grad()
        # pc: (B, 81920, 3)
        # feats: (B, 81920, 3)
        # labels: (B, (voxel_resolution+1)^3)

        pc, feats, labels = data
        pc = pc.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        latent, posterior = model.encode(pc, feats)

        # logits_total, positions_total: length = (voxel_resolution+1)^3 / chunk_size
        # each logits in logits total are (B, 50000)
        # each position in positions total are (B, 50000, 3) <- may not be useful here
        logits_total, positions_total, chuck_size = model.get_mesh_logits(latent, voxel_resolution=voxel_resolution)

        start = 0 # keep track of the chuck
        bce_loss = 0
        for i in range(len(logits_total)):
            # gt: (B, 50000)
            gt = labels[:, start: start + chuck_size]
            logits = logits_total[i]
            bce_loss += bce(logits, gt)
            start += chuck_size

        # Magic number from the paper replaced by kl_scale
        loss = bce_loss + kl_scale * kl_regularization(posterior.mean, posterior.logvar)
        loss.backward()
        optimizer.step()
        print(model.encoder.proj_in.weight)

    validation_losses = []
    with torch.no_grad():
        for data in val_loader:
            pc, feats, labels = data
            pc = pc.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            latent, posterior = model.encode(pc, feats)

            logits_total, positions_total, chuck_size = model.get_mesh_logits(latent, voxel_resolution=voxel_resolution)

            start = 0  # keep track of the chuck
            bce_loss = 0
            for i in range(len(logits_total)):
                gt = labels[:, start: start + chuck_size]
                logits = logits_total[i]
                bce_loss += bce(logits, gt)
                start += chuck_size

            loss = bce_loss + kl_scale * kl_regularization(posterior.mean, posterior.logvar)
            validation_losses.append(loss.detach().cpu().item())
    print(f"validation losses: {np.mean(validation_losses)}")

def train(model, train_loader, val_loader, optimizer, epochs, bce, kl, kl_scale, device, voxel_resolution):
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, val_loader, optimizer, bce, kl, kl_scale, device, voxel_resolution)
        break

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # preparing VAE model
    model_vae = load_pretrained_vae().to(device)
    model_vae.encoder.train()
    model_vae.pre_latent.train()
    model_vae.post_latent.train()
    model_vae.decoder.eval()

    optimizer = torch.optim.AdamW(
        itertools.chain(
            model_vae.encoder.parameters(),
            model_vae.pre_latent.parameters(),
            model_vae.post_latent.parameters()
        ),
        lr=args.lr
    )

    # Preparing dataset
    # download_objaverse(args.download_ratio)  # Commented out as in the original code

    ratio = args.ratio
    dataset = load_dataset("allenai/objaverse-xl")
    dataset = dataset['train'].train_test_split(test_size=ratio)
    train_test_split = dataset['test'].train_test_split(test_size=args.test_size_split)
    train_val_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    train_val_dataset = train_val_dataset.train_test_split(test_size=args.train_val_split)
    train_dataset = train_val_dataset['train']
    val_dataset = train_val_dataset['test']
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print(train_dataset[:12].keys())

    # train_dataset = ObjaverseXLDataset(train_dataset[:10])
    # val_dataset = ObjaverseXLDataset(val_dataset[:5])
    # test_dataset = ObjaverseXLDataset(test_dataset[:5])
    train_dataset = DummyDataset(train_dataset[:5], voxel_resolution=args.voxel_resolution)
    val_dataset = DummyDataset(val_dataset[:5], voxel_resolution=args.voxel_resolution)
    test_dataset = DummyDataset(test_dataset[:5], voxel_resolution=args.voxel_resolution)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    bce = nn.BCEWithLogitsLoss()
    kl = kl_regularization
    train(
        model_vae,
        train_loader,
        val_loader,
        optimizer,
        epochs=args.epochs,
        bce=bce,
        kl=kl,
        kl_scale=args.kl_scale,
        device=device,
        voxel_resolution=args.voxel_resolution
    )

if __name__ == "__main__":
    main()
