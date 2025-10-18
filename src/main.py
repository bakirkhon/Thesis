#import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig # used to load config yaml files
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint # used to save the best weights
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

#from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning) # ignore warnings of this category


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    # else:
    #     model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    # else:
    #     model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model

def get_resume_inference(cfg, model_kwargs):
    """
    Loads a model for inference using the checkpoint path in cfg.general.inference_only.
    Unlike test_only, this does not run Trainer.test() — it lets us call predict_edges().
    """
    saved_cfg = cfg.copy()
    ckpt_path = cfg.general.inference
    name = cfg.general.name + '_inference'

    # Load model from checkpoint
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            ckpt_path,
            # strict=False,
            # map_location="cpu",
            **model_kwargs
        )
        model = model.to("cpu")

    cfg = model.cfg  # Restore training configuration from checkpoint
    cfg.general.inference = ckpt_path
    cfg.general.name = name

    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # print("Select mode:")
    # print("1 - Training")
    # print("2 - Inference (predict edges)")
    # mode = input("Enter 1 or 2: ").strip()

    dataset_config = cfg["dataset"] # load dataset name from config file

    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'famipacking']:
        from datasets.famipacking_dataset import FamipackingGraphDataModule, FamipackingDatasetInfo
        #from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import FamipackingSamplingMetrics #, PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization
      
        # create a data module for train/val/test sets with graph size, node and edge types distributions
        if dataset_config['name']=='famipacking':
            datamodule=FamipackingGraphDataModule(cfg)
        #else:
        #    datamodule=SpectreGraphDataModule(cfg)

        # defines metrics for each dataset
        if dataset_config['name'] == 'famipacking':
            sampling_metrics = FamipackingSamplingMetrics(datamodule)
        #elif dataset_config['name'] == 'sbm':
        #    sampling_metrics = SBMSamplingMetrics(datamodule)
        #elif dataset_config['name'] == 'comm20':
        #   sampling_metrics = Comm20SamplingMetrics(datamodule)
        #else:
        #    sampling_metrics = PlanarSamplingMetrics(datamodule)

        # collects dataset info (info of nodes might be redundant and cause issues)
        dataset_infos = FamipackingDatasetInfo(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() #if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else: #this is our case
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        # key word arguments dictionary that collects all parameters
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    # not relevant
    # elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
    #     from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
    #     from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    #     from diffusion.extra_features_molecular import ExtraMolecularFeatures
    #     from analysis.visualization import MolecularVisualization

    #     if dataset_config["name"] == 'qm9':
    #         from datasets import qm9_dataset
    #         datamodule = qm9_dataset.QM9DataModule(cfg)
    #         dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    #         train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
    #                                                     dataset_infos=dataset_infos, evaluate_dataset=False)
    #     elif dataset_config['name'] == 'guacamol':
    #         from datasets import guacamol_dataset
    #         datamodule = guacamol_dataset.GuacamolDataModule(cfg)
    #         dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
    #         train_smiles = None

    #     elif dataset_config.name == 'moses':
    #         from datasets import moses_dataset
    #         datamodule = moses_dataset.MosesDataModule(cfg)
    #         dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
    #         train_smiles = None
    #     else:
    #         raise ValueError("Dataset not implemented")

        # if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        #     extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        #     # domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        # else:
        #     extra_features = DummyExtraFeatures()
        #     domain_features = DummyExtraFeatures()

        # dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
        #                                         domain_features=domain_features)

        # if cfg.model.type == 'discrete':
        #     train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        # else:
        #     train_metrics = TrainMolecularMetrics(dataset_infos)

        # # We do not evaluate novelty during training
        # sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        # visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        # model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
        #                 'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
        #                 'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])
    elif cfg.general.inference:
        # Inference mode
        cfg, model = get_resume_inference(cfg, model_kwargs)
        os.chdir(cfg.general.inference.split('checkpoints')[0])

        print("Running inference mode (predict_edges)...")
        # Load inference dataset
        inference_path = "/home/bakirkhon/Thesis/3D-bin-packing-master/dataset/inference_dataset_small.pt"
        assert os.path.exists(inference_path), f"File not found: {inference_path}"
        all_graphs = torch.load(inference_path)
        print(f"Loaded {len(all_graphs)} graphs from inference dataset.")

        # Prepare model
        model.eval()
        device = "cpu"
        model.to(device)

        predictions = []

        # Inference loop
        for idx, graph in enumerate(all_graphs):
            X = torch.tensor(graph["X"], dtype=torch.float32).unsqueeze(0).to(device)
            print(f"Processing graph {idx + 1}/{len(all_graphs)} | X shape: {X.shape}")

            E_pred = model.predict_edges(X)  # returns E tensor
            predictions.append({"X": graph["X"], "E": E_pred.cpu().numpy()})

        # Save predictions
        output_file = "/home/bakirkhon/Thesis/outputs/2025-10-14/06-44-43-graph-tf-model/predictions/inference_predictions.pt"
        torch.save(predictions, output_file)
        print(f"✅ Saved predictions for {len(predictions)} graphs to:\n{os.path.abspath(output_file)}")

        return  # Exit after inference
    # elif cfg.general.inference:
    #     # Load checkpoint for inference
    #     cfg, model = get_resume_inference(cfg, model_kwargs)
    #     os.chdir(cfg.general.inference.split('checkpoints')[0])
    #     print("Running inference mode (predict_edges)...")
    #     X_fixed = torch.tensor([[15, 21,  6],
    #                             [21, 23, 11],
    #                             [18, 20, 17],
    #                             [15, 22, 19],
    #                             [16, 24, 17],
    #                             [19, 23,  9],
    #                             [15, 20, 19],
    #                             [22, 24, 11],
    #                             [17, 20, 19],
    #                             [22, 20,  9],
    #                             [40, 40, 30],
    #                             [30, 30, 30]])

    #     if X_fixed.ndim == 2:
    #         X_fixed = X_fixed.unsqueeze(0)

    #     model.eval()
    #     model.to("cpu")
    #     model.predict_edges(X_fixed)
    #     return  # Exit after inference    



    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    # else:
    #     model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val_epoch/E_CE',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="auto",#"ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)

    elif not cfg.general.inference:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()