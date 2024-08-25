import os
import pandas as pd
import numpy as np
import hydra
import logging
from omegaconf import DictConfig,OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="ml.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    #config = OmegaConf.merge(config, {'path_to_save':output_dir})
    ml = instantiate(config['ml']['class_name'], config)
    ml.processing(output_dir)
    '''
    output_dir = hydra_cfg['runtime']['output_dir']
    saving_df_path = os.path.join(output_dir, "feature_dataset.csv")
    df.to_csv(saving_df_path)
    for i in range(len(data)):
        data[i].to_csv(os.path.join(output_dir, data_name[i] + "_feature.csv"))
    '''


if __name__ == "__main__":
    main()