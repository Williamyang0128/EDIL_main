import os
import yaml
import pickle
import torch
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import repeat
from shadow_act.utils.utils import (
    set_seed,
    load_data,
    compute_dict_mean,
)
from shadow_act.network.policy import MMArmPolicy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RmActTrainer:
    def __init__(self, config):
        """
        Initialize trainer, set random seed, load data, save data statistics.
        """
        self._config = config
        self._num_steps = config["num_steps"]
        self._ckpt_dir = config["checkpoint_dir"]
        self._state_dim = config["state_dim"]
        self._real_robot = config["real_robot"]
        self._policy_class = config["policy_class"]
        self._onscreen_render = config["onscreen_render"]
        self._policy_config = config["policy_config"]
        self._camera_names = config["camera_names"]
        self._max_timesteps = config["episode_len"]
        self._task_name = config["task_name"]
        self._temporal_agg = config["policy_config"].get("temporal_agg", False)
        self._onscreen_cam = "angle"
        self._vq = config["policy_config"]["vq"]
        self._batch_size = config["batch_size"]

        self._seed = config["seed"]
        self._eval_every = config["eval_every"]
        self._validate_every = config["validate_every"]
        self._save_every = config["save_every"]
        self._load_pretrain = config["load_pretrain"]
        self._resume_ckpt_path = config["resume_ckpt_path"]

        if config["name_filter"] is None:
            name_filter = lambda n : True
        else:
            name_filter = config["name_filter"]

        # Load data
        self._train_dataloader, self._val_dataloader, self._stats, _ = load_data(
            config["dataset_dir"],
            name_filter,
            self._camera_names,
            self._batch_size,
            self._batch_size,
            config["chunk_size"],
            config["skip_mirrored_data"],
            self._load_pretrain,
            self._policy_class,
            config["stats_dir"],
            config["sample_weights"],
            config["train_ratio"],
        )
        
        # Save data statistics
        if not os.path.exists(self._ckpt_dir):
            os.makedirs(self._ckpt_dir, exist_ok=True)
            
        stats_path = os.path.join(self._ckpt_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(self._stats, f)
        
        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self._ckpt_dir, 'tensorboard_logs'))

    def _make_policy(self):
        """
        Create policy object based on policy class and configuration
        """
        if self._policy_class == "MMArm":
            return MMArmPolicy(self._policy_config)
        else:
            raise NotImplementedError(f"Policy class {self._policy_class} is not implemented. Only 'MMArm' is supported.")

    def _make_optimizer(self):
        """
        Create optimizer based on policy class
        """
        if self._policy_class == "MMArm":
            optimizer = self._policy.configure_optimizers()
        else:
            # TODO: Default to Adam optimizer
            print('Warning: Using default optimizer')
            optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._policy_config["lr"])
        return optimizer

    def _forward_pass(self, data):
        """
        Forward pass, compute loss
        """
        image_data, qpos_data, action_data, is_pad = data
        try:
            image_data, qpos_data, action_data, is_pad = (
                image_data.cuda(),
                qpos_data.cuda(),
                action_data.cuda(),
                is_pad.cuda(),
            )
        except RuntimeError as e:
            logging.error(f"CUDA error: {e}")
            raise
        return self._policy(qpos_data, image_data, action_data, is_pad)

    def _repeater(self):
        """
        Data loader repeater, yield data
        """
        epoch = 0
        for loader in repeat(self._train_dataloader):
            for data in loader:
                yield data
            logging.info(f"Epoch {epoch} done")
            epoch += 1

    def train(self):
        """
        Train model, save best model
        """
        set_seed(self._seed)
        self._policy = self._make_policy()
        min_val_loss = np.inf
        best_ckpt_info = None

        # Load pre-trained model
        if self._load_pretrain:
            # Use configuration for pretrain path instead of hardcoded path
            pretrain_path = self._config.get("pretrain_path", None)
            if pretrain_path and os.path.exists(pretrain_path):
                try:
                    loading_status = self._policy.deserialize(
                        torch.load(pretrain_path)
                    )
                    logging.info(f"loaded! {loading_status}")
                except Exception as e:
                    logging.error(f"Error loading pretrain model from {pretrain_path}: {e}")
            else:
                 logging.warning(f"Pretrain enabled but path not found or invalid: {pretrain_path}")

        # Resume checkpoint
        if self._resume_ckpt_path is not None:
            try:
                loading_status = self._policy.deserialize(torch.load(self._resume_ckpt_path))
                logging.info(f"Resume policy from: {self._resume_ckpt_path}, Status: {loading_status}")
            except FileNotFoundError as e:
                logging.error(f"Checkpoint not found: {e}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")

        self._policy.cuda()

        self._optimizer = self._make_optimizer()
        train_dataloader = self._repeater() 

        # Record training and validation loss
        train_losses = []
        val_losses = []

        for step in tqdm(range(self._num_steps + 1)):
            # Validate model
            if step % self._validate_every != 0:
                pass 
            else:
                logging.info("validating")
                with torch.inference_mode():
                    self._policy.eval()
                    validation_dicts = []
                    for batch_idx, data in enumerate(self._val_dataloader):
                        forward_dict = self._forward_pass(data) 
                        validation_dicts.append(forward_dict)
                        if batch_idx > 50: # Limit validation batches
                            break
    
                    validation_summary = compute_dict_mean(validation_dicts)
                    epoch_val_loss = validation_summary["loss"]
                    val_losses.append(epoch_val_loss.item())  # Record validation loss
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt_info = (
                            step,
                            min_val_loss,
                            deepcopy(self._policy.serialize()),
                        )
    
                logging.info(f"Val loss:   {epoch_val_loss:.5f}")
                summary_string = " ".join(f"{k}: {v.item():.3f}" for k, v in validation_summary.items())
                logging.info(summary_string)
    
                # Use TensorBoard to record validation loss
                self.writer.add_scalar('Validation Loss', epoch_val_loss.item(), step)

            # Train model
            self._policy.train()
            self._optimizer.zero_grad()
            data = next(train_dataloader)
            forward_dict = self._forward_pass(data)
            loss = forward_dict["loss"]
            loss.backward()
            self._optimizer.step()
            train_losses.append(loss.item())  # Record training loss

            # Use TensorBoard to record training loss
            self.writer.add_scalar('Training Loss', loss.item(), step)

            # Save checkpoint
            if step % self._save_every == 0:
                ckpt_path = os.path.join(self._ckpt_dir, f"policy_step_{step}_seed_{self._seed}.ckpt")
                torch.save(self._policy.serialize(), ckpt_path)

        # Save final model
        ckpt_path = os.path.join(self._ckpt_dir, "policy_last.ckpt")
        torch.save(self._policy.serialize(), ckpt_path)

        if best_ckpt_info:
            best_step, min_val_loss, best_state_dict = best_ckpt_info
            ckpt_path = os.path.join(self._ckpt_dir, f"policy_best_step_{best_step}_seed_{self._seed}.ckpt")
            torch.save(best_state_dict, ckpt_path)
            logging.info(f"Training finished:\nSeed {self._seed}, val loss {min_val_loss:.6f} at step {best_step}")

        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig(os.path.join(self._ckpt_dir, 'loss_curve.png'))
        plt.close()

        # Close TensorBoard SummaryWriter
        self.writer.close()

        return best_ckpt_info
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = RmActTrainer(config)
    trainer.train()
