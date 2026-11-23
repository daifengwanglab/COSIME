"""Load and align omics data by sample ID."""
import numpy as np
import pandas as pd
from config import resolve

def load_tables(cfg):
    """Read the views.

    Single view and early fusion: views are aligned by sample ID (the same samples in every view);
    returns X (list per view), y, ids, feature names.
    Late fusion: each view keeps its own samples and outcome; y and ids are lists per view.
    """
    data_cfg = cfg['data']
    target = data_cfg.get('outcome_column', 'y')
    id_column = data_cfg.get('id_column')
    late = data_cfg.get('mode') == 'multi' and cfg.get('model', {}).get('fusion', 'early') == 'late'
    frames = []
    for m in data_cfg['modalities']:
        path = resolve(m['path'])
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if id_column:
            if id_column not in df:
                raise ValueError(f'{path}: missing ID column {id_column}')
            if df[id_column].duplicated().any() or df[id_column].isna().any():
                raise ValueError(f'{path}: IDs must be nonmissing and unique')
            df = df.set_index(id_column)
        else:
            # By default the first column holds the sample IDs.
            if data_cfg.get('first_column_is_id', True):
                df = df.set_index(df.columns[0])
            else:
                df.index = pd.RangeIndex(len(df))
        if df.index.duplicated().any():
            raise ValueError(f'{path}: duplicate sample IDs')
        if target not in df:
            raise ValueError(f'{path}: missing outcome column {target}')
        frames.append(df)
    if late:
        return late_views(frames, target, cfg)
    shared = frames[0].index
    for df in frames[1:]:
        shared = shared.intersection(df.index, sort=False)
    if len(shared) < 10:
        raise ValueError('Too few samples shared across modalities')
    aligned = [df.loc[shared] for df in frames]
    y = pd.to_numeric(aligned[0][target], errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError('Outcome has missing/nonfinite values')
    for df in aligned[1:]:
        other = pd.to_numeric(df[target], errors='raise').to_numpy(dtype=np.float32)
        if not np.array_equal(y, other):
            raise ValueError('Outcomes disagree across modalities for aligned subjects')
    if cfg.get('model', {}).get('type') == 'binary' and not set(np.unique(y)).issubset({0., 1.}):
        raise ValueError('Binary outcomes must be coded 0/1')
    X, names = zip(*(features_of(df, target) for df in aligned))
    return list(X), y, np.asarray(shared).astype(str), list(names)


def features_of(df, target):
    features = df.drop(columns=[target])
    if features.shape[1] < 2:
        raise ValueError('Each modality needs at least two features')
    values = features.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Feature matrices contain missing/nonfinite values')
    return values, list(features.columns)


def late_views(frames, target, cfg):
    """Late fusion: every view with its own samples and outcome."""
    X, ys, ids, names = [], [], [], []
    for j, df in enumerate(frames, start=1):
        if len(df) < 10:
            raise ValueError(f'View {j}: too few samples')
        y = pd.to_numeric(df[target], errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError(f'View {j}: outcome has missing/nonfinite values')
        if cfg.get('model', {}).get('type') == 'binary' and not set(np.unique(y)).issubset({0., 1.}):
            raise ValueError('Binary outcomes must be coded 0/1')
        values, cols = features_of(df, target)
        X.append(values); ys.append(y); ids.append(np.asarray(df.index).astype(str)); names.append(cols)
    return X, ys, ids, names
