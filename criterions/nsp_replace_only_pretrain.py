import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion

@register_criterion("nsp_replace_only_pretrain")
class NSPReplaceOnlyPretrainCriterion(BaseCriterion):
    def __init__(self):
        super(NSPReplaceOnlyPretrainCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, model, batch, device, return_predicts=False):
        inputs = batch["input"] #[batch, n_electrodes, embed]
        masked_input = inputs.to(device) 
        pad_mask = batch["attn_mask"].to(device) 
        coords = batch["coords"].to(device)
        seq_id = batch["seq_id"]
        replace_label = batch["replace_label"]
        position = (coords, seq_id)
        output, (cls_output, token_cls_output) = model.forward(masked_input, pad_mask, position)

        replace_label = (replace_label).type(torch.FloatTensor).to(device)
        replace_label = replace_label[:,1:]
        replace_label = replace_label.flatten()

        replace_outputs = token_cls_output.flatten()
        replace_outputs = replace_outputs[(replace_label==1) | (replace_label==2)]
        replace_label = replace_label[(replace_label==1) | (replace_label==2)]
        replace_label[replace_label==2] = 0
        replace_loss = self.bce_loss(replace_outputs.squeeze(), replace_label)

        #target is [batch, channel, d_embed]
        target = batch["target"].to(device)

        labels = torch.FloatTensor(batch["labels"]).to(device)
        cls_loss = self.bce_loss(cls_output.squeeze(), labels)

        #loss = cls_loss
        loss = cls_loss + replace_loss

        logging_output = {"loss": loss.item(), 
                          "cls_loss": cls_loss.item(),
                          "replace_loss": replace_loss.item(),
                          #"seeds": batch["seeds"]
                         }
        return loss, logging_output



