import yaml
import hydra
from pathlib import Path

def get_multirun_swept_overrides(out_dir:Path):
    if 'MULTIRUN' in hydra.core.hydra_config.HydraConfig.get().overrides.hydra[0]:
        multirun_config = yaml.safe_load((out_dir/'../multirun.yaml').read_text())
        multirun_overrides = multirun_config['hydra']['overrides']['task']
        return [override.split('=')[0] for override in multirun_overrides if ',' in override]
    else:
        return []
