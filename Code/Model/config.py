"""YAML loading and configuration validation."""
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
def resolve(path):
    path=Path(path).expanduser()
    return path if path.is_absolute() else ROOT/path
def get_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError('YAML must contain a mapping')
    data = cfg.get('data', {})
    mode = data.get('mode')
    if mode not in ('single', 'multi'):
        raise ValueError('data.mode must be single or multi')
    modalities = data.get('modalities', [])
    if not isinstance(modalities, list) or len(modalities) < (1 if mode == 'single' else 2):
        raise ValueError('single requires one modality; multi requires at least two')
    if mode == 'single' and len(modalities) != 1:
        raise ValueError('single requires exactly one modality')
    if any(not isinstance(m, dict) or not m.get('path') for m in modalities):
        raise ValueError('Every modality must have a path')
    model = cfg.get('model', {})
    if model.get('type', 'continuous') not in ('continuous', 'binary'):
        raise ValueError('model.type must be continuous or binary')
    if model.get('fusion', 'early') not in ('early', 'late'):
        raise ValueError('model.fusion must be early or late')
    if model.get('predictor', 'regression') not in ('regression', 'NN'):
        raise ValueError('model.predictor must be regression or NN')
    if int(cfg.get('training', {}).get('splits', 5)) < 2:
        raise ValueError('training.splits must be >= 2')
    return cfg
