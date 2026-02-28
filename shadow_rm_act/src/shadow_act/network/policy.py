import torch
import logging
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from shadow_act.models.detr_vae import build_vae

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MMArmPolicy(nn.Module):
    def __init__(self, arm_config):
        """
        Initialize MMArmPolicy class

        Args:
            arm_config (dict): Configuration dictionary
        """
        super().__init__()
        lr_backbone = arm_config["lr_backbone"]
        vq = arm_config["vq"]
        lr = arm_config["lr"]
        weight_decay = arm_config["weight_decay"]

        model = build_vae(
            arm_config["hidden_dim"],
            arm_config["state_dim"],
            arm_config["position_embedding"],
            lr_backbone,
            arm_config["masks"],
            arm_config["backbone"],
            arm_config["dilation"],
            arm_config["dropout"],
            arm_config["nheads"],
            arm_config["dim_feedforward"],
            arm_config["enc_layers"],
            arm_config["dec_layers"],
            arm_config["pre_norm"],
            arm_config["num_queries"],
            arm_config["camera_names"],
            vq,
            arm_config["vq_class"],
            arm_config["vq_dim"],
            arm_config["action_dim"],
            arm_config["no_encoder"],
        )
        model.cuda()

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_dicts, lr=lr, weight_decay=weight_decay
        )
        self.model = model  # CVAE decoder
        self.kl_weight = arm_config["kl_weight"]
        self.vq = vq
        
        # Temporal Aggregation
        self.temporal_agg = arm_config.get("temporal_agg", False)
        self.query_frequency = arm_config.get("query_frequency", 1)
        self.num_queries = arm_config["num_queries"]
        self.all_time_actions = []

        logger.info(f"KL Weight {self.kl_weight}")

    def reset(self):
        """
        Reset temporal aggregation buffers
        """
        self.all_time_actions = []

    def temporal_ensembling(self, all_actions):
        """
        Apply Temporal Ensembling to smooth actions
        
        Args:
            all_actions (torch.Tensor): Predicted action chunk [batch, seq_len, action_dim]
            
        Returns:
            torch.Tensor: Aggregated action for current step [batch, action_dim]
        """
        # Move to CPU numpy for easier handling
        all_actions = all_actions.detach().cpu().numpy()
        self.all_time_actions.append(all_actions)
        
        # Exponential weighting scheme (k=0.01) as per MMArm
        k = 0.01
        exp_weights = []
        actions_populated = []
        
        # Current timestep t relative to the start of the episode
        t = len(self.all_time_actions) - 1
        
        # Collect relevant predictions from past queries
        for i, query_action in enumerate(self.all_time_actions):
            # i is the start time of the query
            # We want the action at time t
            time_offset = t - i
            
            if 0 <= time_offset < self.num_queries:
                # This query covers the current timestep
                actions_populated.append(query_action[:, time_offset, :])
                exp_weights.append(np.exp(-k * time_offset))

        if not actions_populated:
             # Should not happen if logic is correct
             return torch.from_numpy(all_actions[:, 0, :]).float().cuda()
             
        actions_populated = np.stack(actions_populated, axis=0) # [num_overlap, batch, dim]
        exp_weights = np.array(exp_weights) # [num_overlap]
        exp_weights = exp_weights[:, None, None] # Broadcast to match actions [num_overlap, 1, 1]
        
        # Weighted average
        avg_action = np.sum(actions_populated * exp_weights, axis=0) / np.sum(exp_weights, axis=0)
        
        return torch.from_numpy(avg_action).float().cuda() # Assuming CUDA

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        """
        Forward pass

        Args:
            qpos (torch.Tensor): Position tensor
            image (torch.Tensor): Image tensor
            actions (torch.Tensor, optional): Action tensor
            is_pad (torch.Tensor, optional): Padding tensor
            vq_sample (torch.Tensor, optional): VQ sample

        Returns:
            dict: Loss dictionary (during training)
            torch.Tensor: Action tensor (during inference)
        """
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # Training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            loss_dict = dict()
            # The encoder should only condition on actions and proprioception (no vision)
            # as per the text description: q(z | A, s_arm)
            # We explicitly pass None for visual features if the underlying model structure permits,
            # but DETRVAE's encode method interface in detr_vae.py currently doesn't take images anyway.
            # It only takes qpos (s_arm) and actions (A).
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(
                qpos, image, env_state, actions, is_pad
            )
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict["vq_discrepancy"] = F.l1_loss(
                    probs, binaries, reduction="mean"
                )
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            
            # Reconstruction loss (Maximize Log-Likelihood)
            # Assuming Laplace distribution: log p(x) ~ -|x - \hat{x}|
            log_likelihood = -l1 
            
            # Evidence Lower Bound (ELBO)
            # ELBO = E[log p(x|z)] - beta * KL(q(z|x)||p(z))
            elbo = log_likelihood - self.kl_weight * total_kld[0]
            
            # Minimize Negative ELBO
            loss = -elbo

            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss
            return loss_dict
        else:  # Inference time
            a_hat, _, (_, _), _, _ = self.model(
                qpos, image, env_state, vq_sample=vq_sample
            )  # no action, sample from prior

            if self.temporal_agg:
                return self.temporal_ensembling(a_hat)
            
            return a_hat

    def configure_optimizers(self):
        """
        Configure optimizer

        Returns:
            optimizer: Configured optimizer
        """
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        """
        VQ encoding

        Args:
            qpos (torch.Tensor): Position tensor
            actions (torch.Tensor): Action tensor
            is_pad (torch.Tensor): Padding tensor

        Returns:
            torch.Tensor: Binary codes
        """
        actions = actions[:, : self.model.num_queries]
        is_pad = is_pad[:, : self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries

    def serialize(self):
        """
        Serialize model

        Returns:
            dict: Model state dictionary
        """
        return self.state_dict()

    def deserialize(self, model_dict):
        """
        Deserialize model

        Args:
            model_dict (dict): Model state dictionary

        Returns:
            status: Loading status
        """
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    """
    Compute KL divergence

    Args:
        mu (torch.Tensor): Mean tensor
        logvar (torch.Tensor): Log variance tensor

    Returns:
        tuple: Total KL divergence, dimension-wise KL divergence, mean KL divergence
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
