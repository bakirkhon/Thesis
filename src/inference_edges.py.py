import torch
import os
from diffusion_model_discrete import DiscreteDenoisingDiffusion
import utils


@torch.no_grad()
def infer_edges(model, X_fixed, device='cuda', number_chain_steps=50):
    """
    Generate an edge matrix prediction for a given node matrix X_fixed
    using a trained discrete diffusion model (edge-only version).
    """
    model.eval()
    model.to(device)
    X_fixed = X_fixed.to(device).float()

    n_nodes = torch.tensor([X_fixed.size(1)], device=device)
 
    n_max = n_nodes.max().item()
    arange = torch.arange(n_max, device=device).unsqueeze(0)
    node_mask = arange < n_nodes.unsqueeze(1)

    # Run reverse diffusion (sampling)
    samples = model.sample_batch(
        batch_id=0,
        batch_size=1,
        keep_chain=0,
        number_chain_steps=number_chain_steps,
        save_final=1,
        X_fixed=X_fixed,
        num_nodes=n_nodes
    )

    _, E_pred = samples[0]
    return E_pred


def load_model(ckpt_path, dataset_infos, train_metrics, sampling_metrics, extra_features, domain_features):
    """
    Loads a trained diffusion model from a checkpoint.
    """
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        ckpt_path,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=None,
        extra_features=extra_features,
        domain_features=domain_features
    )
    return model


if __name__ == "__main__":
    # --- CONFIG --- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = "Thesis/outputs/2025-10-08/06-41-27-graph-tf-model/checkpoints/graph-tf-model/epoch=299.ckpt"

    # You should load the dataset info / configs used during training
    # If you saved them, import or recreate them here:
    from datasets.famipacking_dataset import FamipackingDatasetInfo, FamipackingGraphDataModule
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/config.yaml")  # or your experiment config
    datamodule = FamipackingGraphDataModule(cfg)
    dataset_infos = FamipackingDatasetInfo(datamodule, cfg.dataset)
    extra_features = utils.DummyExtraFeatures()
    domain_features = utils.DummyExtraFeatures()

    # Dummy metric objects (not needed for inference)
    from metrics.train_metrics import TrainLossDiscrete
    from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = None

    # --- LOAD MODEL --- #
    model = load_model(ckpt_path, dataset_infos, train_metrics, sampling_metrics, extra_features, domain_features)

    # --- PREPARE NODE MATRIX --- #
    X_fixed = torch.tensor([
        [21, 22, 14],
        [22, 20, 8],
        [23, 22, 11],
        [15, 27, 11],
        [21, 29, 14]
    ], dtype=torch.float).unsqueeze(0)  # shape (1, n, 3)

    # --- INFERENCE --- #
    E_pred = infer_edges(model, X_fixed, device=device)
    print("Predicted edge matrix shape:", E_pred.shape)
    print("Predicted edges:", E_pred)
