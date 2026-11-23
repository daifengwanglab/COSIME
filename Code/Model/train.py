"""K-fold COSIME training and holdout evaluation."""
import copy,json,random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from tqdm.auto import tqdm
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from config import resolve
from data_loader import load_tables
from model_loader import load_model
from loss import weighted_loss
from evaluate_holdout import evaluate, evaluate_late

def processed(raw, scalers):
    return scalers['standard_scaler'].transform(scalers['minmax_scaler'].transform(raw)).astype(np.float32)


def write_fi_input(tables, ids, output):
    """Write the processed holdout data used by the explainer (FI_holdout_processed.csv).

    One view: features and y, plus FI_holdout_sample_ids.csv. Several views: one CSV with each
    view's sample-ID column and features; with late fusion the views may have different numbers of
    samples, and shorter views are padded with blanks, which the explainer drops.
    """
    path = output / 'FI_holdout_processed.csv'
    if len(tables) == 1:
        tables[0].to_csv(path, index=False)
        pd.DataFrame({'sample_id': ids[0]}).to_csv(output / 'FI_holdout_sample_ids.csv', index=False)
        tqdm.write(f'Saved FI input: {path} ({len(tables[0])} rows)')
        return
    max_rows = max(len(t) for t in tables)
    columns = {}
    for j, (table, view_ids) in enumerate(zip(tables, ids), start=1):
        columns[f'view{j}__sample_id'] = pd.Series(np.asarray(view_ids), dtype='string')
        for name in table.columns.drop('y', errors='ignore'):
            columns[f'view{j}__{name}'] = table[name].reset_index(drop=True)
    pd.DataFrame(columns).reindex(range(max_rows)).to_csv(path, index=False, na_rep='')
    tqdm.write(f'Saved multi-view FI input: {path} ({max_rows} rows)')


def export_fi_holdout(X, ys, ids, feature_names, holdouts, preprocessing, output):
    """Holdout samples of each view (holdouts: one index array per view), preprocessed with the
    scalers of the best fold."""
    tables = []
    for raw, y, rows, scalers, names in zip(X, ys, holdouts, preprocessing, feature_names):
        df = pd.DataFrame(processed(raw[rows], scalers), columns=names)
        df['y'] = y[rows]
        tables.append(df)
    write_fi_input(tables, [np.asarray(v)[rows] for v, rows in zip(ids, holdouts)], output)


def make_optimizer(model, tr, patience):
    # optimizer: adam (decay = L2 penalty added to the gradient) or adamw (decoupled weight decay)
    name = str(tr.get('optimizer', 'adam')).lower()
    if name not in ('adam', 'adamw'):
        raise ValueError('training.optimizer must be adam or adamw')
    opt = torch.optim.AdamW if name == 'adamw' else torch.optim.Adam
    optimizer = opt(model.parameters(), lr=float(tr.get('learning_rate', .0001)), weight_decay=float(tr.get('decay', .001)))
    gamma = float(tr.get('learning_gamma', .99))
    if not 0 < gamma < 1:
        raise ValueError('learning_gamma must be between 0 and 1')
    # Learning rate x lr_factor after lr_patience epochs without improvement
    # (defaults: learning_gamma and earlystop_patience).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(tr.get('lr_factor', gamma)),
                                                           patience=int(tr.get('lr_patience', patience)))
    return optimizer, scheduler


def refit(cfg, X, ys, rows, best_fold, device, batch_size, late):
    """Train a new model on all training samples (rows: one index array per view).

    Scalers are fit on these samples. The model is trained for the best fold's number of epochs
    (the epoch with its lowest validation loss), with that fold's learning rate at each epoch.
    Late fusion: batches of the views are drawn side by side, as in the folds.
    """
    fold, epochs, lrs = best_fold
    if epochs < 1:
        raise ValueError('Best fold has no epochs to refit')
    scaled, preprocessing = [], []
    for x, r in zip(X, rows):
        minmax = MinMaxScaler().fit(x[r])
        standard = StandardScaler().fit(minmax.transform(x[r]))
        scaled.append(standard.transform(minmax.transform(x)).astype(np.float32))
        preprocessing.append({'minmax_scaler': minmax, 'standard_scaler': standard})
    model = load_model([x.shape[1] for x in X], cfg).to(device)
    optimizer, _ = make_optimizer(model, cfg.get('training', {}), int(cfg.get('training', {}).get('earlystop_patience', 40)))
    if late:
        loaders = [DataLoader(TensorDataset(torch.from_numpy(x[r]), torch.from_numpy(y[r])), batch_size=batch_size,
                              shuffle=True, drop_last=len(r) % batch_size == 1) for x, y, r in zip(scaled, ys, rows)]
    else:
        r = rows[0]
        loaders = [DataLoader(TensorDataset(*(torch.from_numpy(x[r]) for x in scaled), torch.from_numpy(ys[0][r])),
                              batch_size=batch_size, shuffle=True, drop_last=len(r) % batch_size == 1)]
    tqdm.write(f'Refit on all training samples ({", ".join(str(len(r)) for r in rows)}) for {epochs} epochs (best fold {fold})')
    for epoch, lr in enumerate(tqdm(lrs, desc='Refit epochs', unit='epoch', leave=False), start=1):
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        for batches in (zip(*loaders) if late else loaders[0]):
            optimizer.zero_grad()
            if late:
                enc, prediction = zip(*(model.view_forward(j, b[0].to(device)) for j, b in enumerate(batches)))
                enc, prediction, labels = list(enc), torch.cat(prediction), torch.cat([b[1] for b in batches]).to(device)
            else:
                enc, prediction = model([t.to(device) for t in batches[:-1]])
                labels = batches[-1].to(device)
            loss, _ = weighted_loss(enc, prediction, labels, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in refit, epoch {epoch}')
            loss.backward()
            optimizer.step()
    return model, scaled, preprocessing


def save_refit(model, cfg, feature_names, preprocessing, best_fold, rows, output):
    fold, epochs, _ = best_fold
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'feature_names': feature_names,
                'preprocessing': preprocessing, 'refit': True, 'selected_fold': fold, 'epochs': epochs,
                'train_indices': rows}, output / 'best_model.pt')
    tqdm.write(f'Saved refit model: {output / "best_model.pt"}')


def save_outputs(output, cfg, histories, summaries, holdout_histories, holdout_predictions, splits):
    pd.DataFrame(histories).to_csv(output / 'history.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'fold_metrics.csv', index=False)
    pd.DataFrame(holdout_histories).to_csv(output / 'holdout_history.csv', index=False)
    pd.DataFrame(holdout_predictions).to_csv(output / 'holdout_predictions.csv', index=False)
    (output / 'run_config.json').write_text(json.dumps(cfg, indent=2))
    print(f'Completed {splits} folds. Results: {output}', flush=True)


def run(cfg):
    tr = cfg.get('training', {})
    seed = int(tr.get('seed', 100))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = tr.get('device', 'auto')
    device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available() else 'cpu' if device_name == 'auto' else device_name)
    X, y, ids, feature_names = load_tables(cfg)
    output = resolve(cfg.get('output', {}).get('directory', 'outputs/COSIME'))
    output.mkdir(parents=True, exist_ok=True)
    if cfg['data'].get('mode') == 'multi' and cfg['model'].get('fusion', 'early') == 'late':
        return run_late(cfg, X, y, ids, feature_names, output, seed, device)
    n = len(y)
    splits = int(tr.get('splits', 5))
    if n < splits + 5:
        raise ValueError('Insufficient subjects for requested folds')
    trainval, holdout = train_test_split(np.arange(n), test_size=float(tr.get('holdout_fraction', .25)), random_state=seed, shuffle=True)
    np.savez(output / 'indices.npz', trainval=trainval, holdout=holdout)
    kfold = KFold(n_splits=splits, shuffle=True, random_state=int(tr.get('cv_seed', 42)))
    batch_size = int(tr.get('batch_size', 32))
    epochs = int(tr.get('epochs', 100))
    patience = int(tr.get('earlystop_patience', 40))
    delta = float(tr.get('delta', .001))
    histories, summaries, holdout_histories, holdout_predictions = [], [], [], []
    overall_best_loss = float('inf')
    best_fold = None
    for fold, (tr_local, val_local) in enumerate(tqdm(kfold.split(trainval), total=splits, desc="CV folds", unit="fold"), start=1):
        train_idx, val_idx = trainval[tr_local], trainval[val_local]
        # Scalers are fit on the fold's training samples only.
        scaled = []
        preprocessing = []
        for j, x in enumerate(X):
            minmax = MinMaxScaler().fit(x[train_idx])
            standard = StandardScaler().fit(minmax.transform(x[train_idx]))
            scaled.append(standard.transform(minmax.transform(x)).astype(np.float32))
            preprocessing.append({'minmax_scaler': minmax, 'standard_scaler': standard})
        model = load_model([x.shape[1] for x in X], cfg).to(device)
        optimizer, scheduler = make_optimizer(model, tr, patience)
        best_loss, best_state, no_improve = float('inf'), None, 0
        best_epoch, lrs = 0, []
        dataset = TensorDataset(*(torch.from_numpy(x[train_idx]) for x in scaled), torch.from_numpy(y[train_idx]))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        if len(train_idx) % batch_size == 1:
            # Avoid BatchNorm batches of size one.
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        if len(train_idx) < 2:
            raise ValueError('Training fold too small for BatchNorm')
        epoch_bar = tqdm(range(1, epochs + 1), desc=f"Fold {fold}/{splits} epochs", unit="epoch", leave=False)
        for epoch in epoch_bar:
            lrs.append(optimizer.param_groups[0]['lr'])
            model.train()
            train_losses = []
            batch_bar = tqdm(loader, desc=f"Fold {fold} epoch {epoch} batches", unit="batch", leave=False)
            for batch in batch_bar:
                features = [t.to(device) for t in batch[:-1]]
                labels = batch[-1].to(device)
                optimizer.zero_grad()
                enc, prediction = model(features)
                loss, prediction_loss = weighted_loss(enc, prediction, labels, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss in fold {fold}, epoch {epoch}')
                loss.backward()
                optimizer.step()
                train_losses.append(prediction_loss)
                batch_bar.set_postfix(loss=f"{float(prediction_loss):.4f}")
            _, val = evaluate(model, scaled, y, val_idx, batch_size, device, cfg)
            val_loss = val['prediction_loss']
            histories.append({'fold': fold, 'epoch': epoch, 'train_prediction_loss': float(np.mean(train_losses)),
                              **{f'val_{k}': v for k, v in val.items()}})
            epoch_bar.set_postfix(train=f'{np.mean(train_losses):.4f}', val=f'{val_loss:.4f}', best=f'{min(best_loss, val_loss):.4f}')
            if val_loss < best_loss - delta or best_state is None:
                best_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
                best_epoch = epoch
            else:
                no_improve += 1
            scheduler.step(val_loss)
            if no_improve >= patience:
                break
        model.load_state_dict(best_state)
        checkpoint = {
            'model_state_dict': best_state,
            'config': cfg,
            'feature_names': feature_names,
            'preprocessing': preprocessing,
            'fold': fold,
            'best_val_loss': best_loss,
            'best_epoch': best_epoch,
            'train_indices': train_idx,
            'validation_indices': val_idx,
        }
        torch.save(checkpoint, output / f'fold_{fold}_best.pt')
        if best_loss < overall_best_loss:
            overall_best_loss, best_fold = best_loss, (fold, best_epoch, lrs[:best_epoch])
            tqdm.write(f'New best fold: fold={fold}, validation_loss={best_loss:.6f}, epoch={best_epoch}')
        _, val = evaluate(model, scaled, y, val_idx, batch_size, device, cfg)
        hold_pred, hold = evaluate(model, scaled, y, holdout, batch_size, device, cfg)
        summaries.append({'fold': fold, 'best_val_loss': best_loss, 'best_epoch': best_epoch, **{f'val_{k}': v for k, v in val.items()},
                          **{f'holdout_{k}': v for k, v in hold.items()}})
        holdout_histories.append({'fold': fold, **hold})
        holdout_predictions.extend({'fold': fold, 'sample_id': ids[idx], 'observed': float(y[idx]), 'prediction': float(pred)} for idx, pred in zip(holdout, hold_pred))
    # Refit on all training samples (everything except the holdout) with the best fold's epochs.
    model, scaled, preprocessing = refit(cfg, X, [y] * len(X), [trainval] * len(X), best_fold, device, batch_size, late=False)
    save_refit(model, cfg, feature_names, preprocessing, best_fold, trainval, output)
    hold_pred, hold = evaluate(model, scaled, y, holdout, batch_size, device, cfg)
    pd.DataFrame([hold]).to_csv(output / 'final_holdout.csv', index=False)
    pd.DataFrame({'sample_id': ids[holdout], 'observed': y[holdout], 'prediction': hold_pred}).to_csv(output / 'final_holdout_predictions.csv', index=False)
    # Holdout data for the explainer, preprocessed with the scalers of the refit model.
    export_fi_holdout(X, [y] * len(X), [ids] * len(X), feature_names, [holdout] * len(X), preprocessing, output)
    save_outputs(output, cfg, histories, summaries, holdout_histories, holdout_predictions, splits)


def late_holdouts(cfg, ys, test_size, seed):
    """Holdout rows of each view: data.holdout_indices (one CSV of row positions per view,
    1-based by default, data.holdout_indices_base), or a random split of each view."""
    files = cfg['data'].get('holdout_indices')
    if files is None:
        return [train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, shuffle=True) for y in ys]
    if len(files) != len(ys):
        raise ValueError('data.holdout_indices: provide one CSV per view')
    base = int(cfg['data'].get('holdout_indices_base', 1))
    out = []
    for j, (f, y) in enumerate(zip(files, ys), start=1):
        rows = pd.to_numeric(pd.read_csv(resolve(f), header=None).iloc[:, 0], errors='raise').to_numpy(dtype=np.int64) - base
        if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(y) or len(np.unique(rows)) != len(rows):
            raise ValueError(f'View {j}: holdout row indices must be unique and within the view')
        out.append((np.setdiff1d(np.arange(len(y)), rows), rows))
    return out


def run_late(cfg, X, ys, ids, feature_names, output, seed, device):
    """Late fusion: each view has its own samples and outcome. Each view gets its own holdout and
    folds (fold k of every view is used together); training batches of the views are drawn side by
    side. Loss: KLD of each view, LOT between the views, and the prediction loss of every view's
    predictions against its own outcomes."""
    tr = cfg.get('training', {})
    splits = int(tr.get('splits', 5))
    if min(len(y) for y in ys) < splits + 5:
        raise ValueError('Insufficient samples in a view for the requested folds')
    split = late_holdouts(cfg, ys, float(tr.get('holdout_fraction', .25)), seed)
    trainvals, holdouts = [s[0] for s in split], [s[1] for s in split]
    np.savez(output / 'indices.npz', **{f'trainval_{j}': t for j, t in enumerate(trainvals, start=1)},
             **{f'holdout_{j}': h for j, h in enumerate(holdouts, start=1)})
    kfold = KFold(n_splits=splits, shuffle=True, random_state=int(tr.get('cv_seed', 42)))
    batch_size = int(tr.get('batch_size', 32))
    epochs = int(tr.get('epochs', 100))
    patience = int(tr.get('earlystop_patience', 40))
    delta = float(tr.get('delta', .001))
    histories, summaries, holdout_histories, holdout_predictions = [], [], [], []
    overall_best_loss = float('inf')
    best_fold = None
    folds = zip(*(kfold.split(t) for t in trainvals))
    for fold, pairs in enumerate(tqdm(folds, total=splits, desc="CV folds", unit="fold"), start=1):
        train_idx = [t[a] for t, (a, _) in zip(trainvals, pairs)]
        val_idx = [t[b] for t, (_, b) in zip(trainvals, pairs)]
        # Scalers are fit on each view's fold training samples only.
        scaled, preprocessing = [], []
        for x, rows in zip(X, train_idx):
            minmax = MinMaxScaler().fit(x[rows])
            standard = StandardScaler().fit(minmax.transform(x[rows]))
            scaled.append(standard.transform(minmax.transform(x)).astype(np.float32))
            preprocessing.append({'minmax_scaler': minmax, 'standard_scaler': standard})
        model = load_model([x.shape[1] for x in X], cfg).to(device)
        optimizer, scheduler = make_optimizer(model, tr, patience)
        best_loss, best_state, no_improve = float('inf'), None, 0
        best_epoch, lrs = 0, []
        if min(len(rows) for rows in train_idx) < 2:
            raise ValueError('Training fold too small for BatchNorm')
        # Avoid BatchNorm batches of size one.
        loaders = [DataLoader(TensorDataset(torch.from_numpy(x[rows]), torch.from_numpy(y[rows])), batch_size=batch_size,
                              shuffle=True, drop_last=len(rows) % batch_size == 1) for x, y, rows in zip(scaled, ys, train_idx)]
        epoch_bar = tqdm(range(1, epochs + 1), desc=f"Fold {fold}/{splits} epochs", unit="epoch", leave=False)
        for epoch in epoch_bar:
            lrs.append(optimizer.param_groups[0]['lr'])
            model.train()
            train_losses = []
            batch_bar = tqdm(zip(*loaders), total=min(len(l) for l in loaders), desc=f"Fold {fold} epoch {epoch} batches",
                             unit="batch", leave=False)
            for batches in batch_bar:
                optimizer.zero_grad()
                enc, prediction = zip(*(model.view_forward(j, b[0].to(device)) for j, b in enumerate(batches)))
                labels = torch.cat([b[1] for b in batches]).to(device)
                loss, prediction_loss = weighted_loss(list(enc), torch.cat(prediction), labels, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss in fold {fold}, epoch {epoch}')
                loss.backward()
                optimizer.step()
                train_losses.append(prediction_loss)
                batch_bar.set_postfix(loss=f"{float(prediction_loss):.4f}")
            _, val = evaluate_late(model, scaled, ys, val_idx, batch_size, device, cfg)
            val_loss = val['prediction_loss']
            histories.append({'fold': fold, 'epoch': epoch, 'train_prediction_loss': float(np.mean(train_losses)),
                              **{f'val_{k}': v for k, v in val.items()}})
            epoch_bar.set_postfix(train=f'{np.mean(train_losses):.4f}', val=f'{val_loss:.4f}', best=f'{min(best_loss, val_loss):.4f}')
            if val_loss < best_loss - delta or best_state is None:
                best_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
                best_epoch = epoch
            else:
                no_improve += 1
            scheduler.step(val_loss)
            if no_improve >= patience:
                break
        model.load_state_dict(best_state)
        checkpoint = {'model_state_dict': best_state, 'config': cfg, 'feature_names': feature_names, 'preprocessing': preprocessing,
                      'fold': fold, 'best_val_loss': best_loss, 'best_epoch': best_epoch, 'train_indices': train_idx, 'validation_indices': val_idx}
        torch.save(checkpoint, output / f'fold_{fold}_best.pt')
        if best_loss < overall_best_loss:
            overall_best_loss, best_fold = best_loss, (fold, best_epoch, lrs[:best_epoch])
            tqdm.write(f'New best fold: fold={fold}, validation_loss={best_loss:.6f}, epoch={best_epoch}')
        _, val = evaluate_late(model, scaled, ys, val_idx, batch_size, device, cfg)
        hold_pred, hold = evaluate_late(model, scaled, ys, holdouts, batch_size, device, cfg)
        summaries.append({'fold': fold, 'best_val_loss': best_loss, 'best_epoch': best_epoch, **{f'val_{k}': v for k, v in val.items()},
                          **{f'holdout_{k}': v for k, v in hold.items()}})
        holdout_histories.append({'fold': fold, **hold})
        for j, (rows, preds) in enumerate(zip(holdouts, hold_pred), start=1):
            holdout_predictions.extend({'fold': fold, 'view': j, 'sample_id': ids[j - 1][i], 'observed': float(ys[j - 1][i]),
                                        'prediction': float(p)} for i, p in zip(rows, preds))
    # Refit on all training samples of each view (everything except the holdout) with the best fold's epochs.
    model, scaled, preprocessing = refit(cfg, X, ys, trainvals, best_fold, device, batch_size, late=True)
    save_refit(model, cfg, feature_names, preprocessing, best_fold, trainvals, output)
    hold_pred, hold = evaluate_late(model, scaled, ys, holdouts, batch_size, device, cfg)
    pd.DataFrame([hold]).to_csv(output / 'final_holdout.csv', index=False)
    pd.DataFrame([{'view': j, 'sample_id': ids[j - 1][i], 'observed': float(ys[j - 1][i]), 'prediction': float(p)}
                  for j, (rows, preds) in enumerate(zip(holdouts, hold_pred), start=1) for i, p in zip(rows, preds)]
                 ).to_csv(output / 'final_holdout_predictions.csv', index=False)
    # Holdout data for the explainer, preprocessed with the scalers of the refit model.
    export_fi_holdout(X, ys, ids, feature_names, holdouts, preprocessing, output)
    save_outputs(output, cfg, histories, summaries, holdout_histories, holdout_predictions, splits)
