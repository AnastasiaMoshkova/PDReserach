import os
import pandas as pd
import numpy as np
import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import instantiate
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="auto_marking.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    auto = instantiate(config['class_name'], config)
    auto.processing(output_dir)
    '''
    if config['error_calculation']:
        output_dir = hydra_cfg['runtime']['output_dir']
        saving_df_path = os.path.join(output_dir, "error.csv")
        df.to_csv(saving_df_path)
    '''
if __name__ == "__main__":
    main()