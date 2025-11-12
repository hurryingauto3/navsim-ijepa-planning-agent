from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch, split: str):
        features, targets = batch
        predictions = self.agent.forward(features)

        loss_dict = self.agent.compute_loss(features, targets, predictions)
        total_loss = loss_dict["loss"]

        logging_prefix = "train" if split == "train" else "val"

        self.log(f"{logging_prefix}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/trajectory_loss", loss_dict["trajectory_loss"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{logging_prefix}/agent_class_loss", loss_dict["agent_class_loss"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{logging_prefix}/agent_box_loss", loss_dict["agent_box_loss"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{logging_prefix}/bev_semantic_loss", loss_dict["bev_semantic_loss"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return total_loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
