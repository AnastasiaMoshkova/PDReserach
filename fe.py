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

@hydra.main(config_path="configs", config_name="fe.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    fe = instantiate(config['feature_extraction']['class_name'], config)
    #df = fe.dataset_processing()
    #df, data, data_name = fe.feature_extraction_dataset(df)
    output_dir = hydra_cfg['runtime']['output_dir']
    fe.processing(output_dir)
    #saving_df_path = os.path.join(output_dir, "feature_dataset.csv")
    #df.to_csv(saving_df_path)
    #for i in range(len(data)):
        #data[i].to_csv(os.path.join(output_dir, data_name[i] + "_feature.csv"))


if __name__ == "__main__":
    main()