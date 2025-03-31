#python3 load_and_save_weights_to_hf.py +model=pt_custom_model

from omegaconf import DictConfig, OmegaConf
import hydra
from models.pt_model_custom import PtModelCustom
import torch

@hydra.main(config_path="conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    model = PtModelCustom()
    model.build_model(cfg.model)

    model_path = '/storage/czw/MultiBrainBERT/outputs/randomized_replacement_no_gaussian_blur.pth' 
    state_dict = torch.load(model_path)['model']

    model.load_state_dict(state_dict) 
    model.save_pretrained("debug_pt_model")

if __name__ == "__main__":
    main()
