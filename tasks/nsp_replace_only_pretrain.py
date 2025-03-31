import logging
import numpy as np
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import nsp_replace_only_pretrain_collator#, spec_target_pretrain_collator
from util.tensorboard_utils import plot_tensorboard_line
from sklearn.metrics import roc_auc_score, f1_score

log = logging.getLogger(__name__)

@register_task(name="nsp_replace_only_pretrain")
class NSPReplaceOnlyPretrainTask(BaseTask):
    def __init__(self, cfg):
        super(NSPReplaceOnlyPretrainTask, self).__init__(cfg)

    def build_model(self, cfg):
        assert hasattr(self, "dataset")
        return models.build_model(cfg)
        
    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss":0, "cls_loss":0, "replace_loss":0}
        predicts, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch["input"] = batch["input"].to(device)
                _, valid_outs = criterion(model, batch, device, return_predicts=True)

                #predicts.append(valid_outs["predicts"])
                labels.append(batch["labels"])
                all_outs["loss"] += valid_outs["loss"]
                all_outs["cls_loss"] += valid_outs["cls_loss"]
                all_outs["replace_loss"] += valid_outs["replace_loss"]
        #labels = np.array([x for y in labels for x in y])
        #predicts = [np.array([p]) if len(p.shape)==0 else p for p in predicts]
        #predicts = np.concatenate(predicts)
        #roc_auc = roc_auc_score(labels, predicts)
        #f1 = f1_score(labels, np.round(predicts))
        all_outs["loss"] /= len(valid_loader)
        all_outs["cls_loss"] /= len(valid_loader)
        all_outs["replace_loss"] /= len(valid_loader)
        #all_outs["roc_auc"] = roc_auc
        #all_outs["f1"] = f1
        return all_outs

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, collate_fn=nsp_replace_only_pretrain_collator, **kwargs)

    def output_logs(self, train_logging_outs, val_logging_outs, writer, global_step):
        val_loss = val_logging_outs["loss"]
        val_cls_loss = val_logging_outs["cls_loss"]
        val_replace_loss = val_logging_outs["replace_loss"]

        train_loss = train_logging_outs["loss"]
        train_cls_loss = train_logging_outs["cls_loss"]
        train_replace_loss = train_logging_outs["replace_loss"]

        if writer is not None:
            writer.add_scalar("val_loss", val_loss, global_step)
            writer.add_scalar("val_cls_loss", val_cls_loss, global_step)
            writer.add_scalar("val_replace_loss", val_replace_loss, global_step)

            writer.add_scalar("train_loss", train_loss, global_step)
            writer.add_scalar("train_cls_loss", train_cls_loss, global_step)
            writer.add_scalar("train_replace_loss", train_replace_loss, global_step)

        log.info(f'val_loss: {val_loss:.6g}, val_cls_loss: {val_cls_loss:.6g}, val_replace_loss {val_replace_loss:.6g}')

        #with open("/storage/czw/MultiBrainBERT/outputs/seeds.json", "w") as f:#TODO hardcode
        #    json.dump(val_logging_outs["seeds"], f)

        #image = train_logging_outs["images"]["wav"]
        #label = train_logging_outs["images"]["wav_label"]
        #tb_image = plot_tensorboard_line(image, title=label)
        #if writer is not None:
        #    writer.add_image("raw_wave", tb_image, global_step)






