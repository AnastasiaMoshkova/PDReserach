import hydra
import os
import logging
from omegaconf import DictConfig
from hydra.utils import instantiate
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="error_am.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    error_auto = instantiate(config['class_name'], config)
    output_dir = hydra_cfg['runtime']['output_dir']
    error_auto.processing(output_dir)
    '''
    if config['error_calculation_save']:
        output_dir = hydra_cfg['runtime']['output_dir']
        saving_df_path = os.path.join(output_dir, "error.csv")
        df.to_csv(saving_df_path)
    '''

if __name__ == "__main__":
    main()