import os
import pandas as pd
import numpy as np
import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="palm_width_calculation.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    ch = instantiate(config['class_name'], config)
    ch.processing()


if __name__ == "__main__":
    main()