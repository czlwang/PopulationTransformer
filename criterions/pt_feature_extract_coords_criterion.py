import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion

#pt stands for population transformer
@register_criterion("pt_feature_extract_coords_criterion")
class PTFeatureExtractCoordsCriterion(BaseCriterion):
    def __init__(self):
        super(PTFeatureExtractCoordsCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg
        self.sigmoid = nn.Sigmoid()
        if 'loss_fn' in cfg and cfg.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, model, batch, device, return_predicts=False):
        #TODO fix the dataset here. 
        inputs = batch["input"].to(device) #potentially don't move to device if dataparallel
        pad_mask = batch["attn_mask"].to(device)
        coords = batch["coords"]
        seq_id = batch["seq_id"]
        position = (coords, seq_id)

        output = model.forward(inputs, pad_mask, position)
        labels = torch.FloatTensor(batch["labels"]).to(output.device)
        output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        if return_predicts:
            if 'loss_fn' in self.cfg and self.cfg.loss_fn == "mse":
                predicts = output.squeeze().detach().cpu().numpy()
            else:
                predicts = self.sigmoid(output).squeeze().detach().cpu().numpy()
            logging_output = {"loss": loss.item(),
                              "predicts": predicts,
                              }
        else:
            logging_output = {"loss": loss.item(),
                              }
        return loss, logging_output

