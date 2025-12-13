import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        # self.val_nll = NLL()
        # self.val_X_kl = SumExceptBatchKL()
        # self.val_E_kl = SumExceptBatchKL()
        # self.val_X_logp = SumExceptBatchMetric()
        # self.val_E_logp = SumExceptBatchMetric()

        # self.test_nll = NLL()
        # self.test_X_kl = SumExceptBatchKL()
        # self.test_E_kl = SumExceptBatchKL()
        # self.test_X_logp = SumExceptBatchMetric()
        # self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':        
            node_types = self.dataset_info.node_types.float() 
            x_marginals = torch.zeros(3) # dummy node_types

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_loss = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        print("Training step")
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=None, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=None, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)
        # loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
        #                        true_X=X, true_E=E, true_y=data.y,
        #                        log=i % self.log_every_steps == 0)
        self.train_metrics(masked_pred_X=None, masked_pred_E=pred.E, true_X=None, true_E=E,
                               log=i % self.log_every_steps == 0)
        # self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
        #                    log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        #self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_fit_end(self):
        """Save model predictions for the entire training dataset after training completes."""
        self.print("Generating predictions for the entire training dataset...")

        all_pred_E = []
        all_true_E = []

        self.eval()
        with torch.no_grad():
            for batch in self.trainer.datamodule.train_dataloader():
                batch = batch.to(self.device)
                dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                dense_data = dense_data.mask(node_mask)

                X, E, y = dense_data.X, dense_data.E, batch.y
                noisy_data = self.apply_noise(X, E, y, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)

                # Predict edge probabilities and convert to one-hot
                pred_E = F.softmax(pred.E, dim=-1)
                E_pred_discrete = F.one_hot(torch.argmax(pred_E, dim=-1),
                                            num_classes=self.Edim_output).float()

                # Mask out padded regions
                mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
                E_pred_discrete = E_pred_discrete * mask.unsqueeze(-1)
                E = E * mask.unsqueeze(-1)

                all_pred_E.append(E_pred_discrete.cpu())
                all_true_E.append(E.cpu())

        # Save as a list - no concatenation
        os.makedirs("train_predictions", exist_ok=True)
        torch.save({
            "predicted_E": all_pred_E,  # list of tensors, each (n_i, n_i, num_edge_classes)
            "true_E": all_true_E
        }, "train_predictions/train_set_E_predictions.pt")

        self.print(f"Saved {len(all_pred_E)} edge predictions for training set "
                f"to train_predictions/train_set_E_predictions.pt")


    # Reset metrics and start the epoch timer before def training_step
    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                   #f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")
        # self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
        #               f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
        #               f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
        #               f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")

    # Reset metrics before def validation_step
    def on_validation_epoch_start(self) -> None:
        self.print("Starting validation epoch...")
        self.train_loss.reset()
        # self.val_nll.reset()
        #self.val_X_kl.reset()
        # self.val_E_kl.reset()
        #self.val_X_logp.reset()
        # self.val_E_logp.reset()
        # self.sampling_metrics.reset()

    def validation_step(self, data, i):
        self.print("Validation step")
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        
        # Compute only CE loss (same as training)
        loss = self.train_loss(masked_pred_X=None, masked_pred_E=pred.E, pred_y=pred.y,
                            true_X=None, true_E=E, true_y=data.y, log=False)
        return {"val_loss": loss}
        # nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        # return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        to_log_val = self.train_loss.log_val_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: val_E_CE: {to_log_val['val_epoch/E_CE'] :.3f} --")
        # self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
        #               f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
        #               f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
        #               f" -- {time.time() - self.start_epoch_time:.1f}s ")
        avg_val_loss = to_log_val['val_epoch/E_CE']
        self.log("val_epoch/E_CE", avg_val_loss)

        # Update best validation loss
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss

        self.print(f"Best Validation CE Loss so far: {self.best_val_loss:.4f}\n")

       
    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.train_loss.reset()

        # self.test_nll.reset()
        # #self.test_X_kl.reset()
        # self.test_E_kl.reset()
        # #self.test_X_logp.reset()
        # self.test_E_logp.reset()
        # if self.local_rank == 0:
        #     utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=None, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=None, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)
        return {"test_loss": loss}

    def on_test_epoch_end(self) -> None:
        to_log_test = self.train_loss.log_test_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: test_E_CE: {to_log_test['test_epoch/E_CE'] :.3f} --")
        
    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)

        # Compute transition probabilities
        #probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        #probE[probE.sum(dim=-1) <= 0] = 1.0 / probE.shape[-1]

        sampled_t = diffusion_utils.sample_discrete_features(probX=None, probE=probE, node_mask=node_mask)

        #X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        X_t = X
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    
    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)
    @torch.no_grad()
    def predict_edges(self, X_fixed: torch.Tensor, number_chain_steps: int = 50):
        """
        Predict edges given fixed node features X_fixed.
        Args:
            X_fixed: (1, n, dx) tensor of node features.
        Returns:
            E_pred: predicted adjacency tensor (1, n, n, de)
        """
        import torch

        X_fixed = X_fixed.to(self.device)
        bs, n_max, _ = X_fixed.shape
        node_mask = torch.ones(bs, n_max, dtype=torch.bool, device=self.device)

        # Start from pure edge noise
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        E, y = z_T.E, z_T.y

        # Reverse diffusion process
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((bs, 1), device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s, _ = self.sample_p_zs_given_zt(s_norm, t_norm, X_fixed, E, y, node_mask)
            E, y = sampled_s.E, sampled_s.y

        # Return the predicted edge tensor
        return E.detach().cpu()

    
    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           If last_step, return the graph prediction as well"""
        bs, n, _, de = E_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        #pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        # p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
        #                                                                                    Qt=Qt.X,
        #                                                                                    Qsb=Qsb.X,
        #                                                                                    Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        # weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        # unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        # unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        # prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(probX=None, probE=prob_E, node_mask=node_mask)
        X_s = X_t   # keep X unchanged
        # sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        # X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
