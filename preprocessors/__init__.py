from .stft import STFTPreprocessor
from .superlet_preprocessor import SuperletPreprocessor
from .wav_preprocessor import WavPreprocessor
from .multi_elec_spec_pretrained import MultiElecSpecPretrained
from .identity_preprocessor import IdentityPreprocessor
__all__ = ["STFTPreprocessor",
           "SuperletPreprocessor",
           "WavPreprocessor",
           "MultiElecSpecPretrained",
           "IdentityPreprocessor"
          ]

def build_preprocessor(preprocessor_cfg):
    if preprocessor_cfg.name == "stft":
        extracter = STFTPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "superlet":
        extracter = SuperletPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "wav_preprocessor":
        extracter = WavPreprocessor(preprocessor_cfg)
    elif preprocessor_cfg.name == "multi_elec_spec_pretrained":
        extracter = MultiElecSpecPretrained(preprocessor_cfg)
    elif preprocessor_cfg.name == "identity_preprocessor":
        extracter = IdentityPreprocessor(preprocessor_cfg)
    else:
        raise ValueError("Specify preprocessor")
    return extracter
