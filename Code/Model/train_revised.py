import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from tqdm import tqdm

from loss import compute_weighted_loss, KL_divergence, LOT
from utils import log_results, EarlyStopper, AUROC

def train_model_binary(model, *datasets, batch_size, learning_rate, m_type, epochs, save_path, splits, device, **kwargs):
    # Prepare DataLoader for batch processing
    train_data = torch.utils.data.TensorDataset(*[torch.tensor(data.values, dtype=torch.float) for data_tuple in datasets for data in data_tuple])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss() if m_type == 'binary' else torch.nn.MSELoss()

    model.train()

    history = {  # Store training history
        'KLD_train_loss_A': [],
        'KLD_train_loss_B': [],
        'OT_train_loss': [],
        'classification_train_loss': [],
        'AUROC_val': [],
        'AUPRC_val': [],
        'accuracy_val': [],
        'KLD_val_loss_A': [],
        'KLD_val_loss_B': [],
        'OT_val_loss': [],
        'classification_val_loss': [],
        'fold_num': [],
        'epoch_num': [],
    }

    holdout_history = {  # Store holdout evaluation history
        'KLD_eval_loss_A': [],
        'KLD_eval_loss_B': [],
        'OT_eval_loss': [],
        'classification_eval_loss': [],
        'AUROC_eval': [],
        'AUPRC_eval': [],
        'accuracy_eval': [],
        'fold_num': [],
    }

    best_model_state_fold = None  # To store the best model state
    best_val_loss = float('inf')  # Initialize best validation loss
    best_predicted_values = None  # To store predicted values from the best model
    best_actual_values = None  # To store actual values from the best model

    for fold, (pair_A, pair_B) in enumerate(zip(splits.split(datasets[0]), splits.split(datasets[1]))):  # Cross-validation loop
        print(f"Fold {fold + 1}")

        (train_idx_A, val_idx_A) = pair_A
        (train_idx_B, val_idx_B) = pair_B

        # Split the data
        data_A_train, data_A_val = datasets[0][train_idx_A], datasets[0][val_idx_A]
        data_B_train, data_B_val = datasets[1][train_idx_B], datasets[1][val_idx_B]

        train_loader_A = DataLoader(torch.utils.data.TensorDataset(data_A_train, data_A_val), batch_size=batch_size, shuffle=True)
        val_loader_A = DataLoader(torch.utils.data.TensorDataset(data_A_val, data_A_val), batch_size=batch_size, shuffle=False)
        train_loader_B = DataLoader(torch.utils.data.TensorDataset(data_B_train, data_B_val), batch_size=batch_size, shuffle=True)
        val_loader_B = DataLoader(torch.utils.data.TensorDataset(data_B_val, data_B_val), batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, scheduler, and early stopping
        input_dims = [data_A_train.shape[1], data_B_train.shape[1]]
        model = Model(*input_dims, dim=kwargs['dim'], dropout=kwargs['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=kwargs['decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=kwargs['learning_gamma'], patience=10, verbose=True)

        early_stopper = EarlyStopper(patience=kwargs['earlystop_patience'], min_delta=kwargs['delta'])

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')

            model.train()

            # Initialize loss accumulators for the epoch
            KLD_loss_A_epoch = 0.0
            KLD_loss_B_epoch = 0.0
            OT_loss_epoch = 0.0
            classification_loss_epoch = 0.0

            total_iterations = len(train_loader_A)  # Total iterations per fold
            for idx, ((data_A_train, labels_A_train), (data_B_train, labels_B_train)) in tqdm(enumerate(zip(train_loader_A, train_loader_B)), total=total_iterations, desc='Training'):
                data_A_train = data_A_train.to(device)
                data_B_train = data_B_train.to(device)
                labels_A_train = labels_A_train.to(device)
                labels_B_train = labels_B_train.to(device)

                optimizer.zero_grad()

                # Forward pass
                nets, logits = model(data_A_train, data_B_train)
                (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets
                logits_A, logits_B = logits

                KLD_loss_A = KL_divergence(mu_A, logsigma_A)
                KLD_loss_B = KL_divergence(mu_B, logsigma_B)
                OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

                # Combine logits and labels for loss calculation
                logits = torch.cat((logits_A, logits_B), dim=0)
                labels = torch.cat((labels_A_train, labels_B_train), dim=0)

                # Classification loss
                classification_loss = criterion(logits.squeeze(), labels.squeeze())

                # Compute the weighted loss using compute_weighted_loss function
                loss = compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, classification_loss,
                                             kwargs['kld_1_weight'], kwargs['kld_2_weight'], kwargs['ot_weight'], kwargs['cl_weight'])

                loss.backward()
                optimizer.step()

                # Accumulate losses for the epoch
                KLD_loss_A_epoch += KLD_loss_A.item()
                KLD_loss_B_epoch += KLD_loss_B.item()
                OT_loss_epoch += OT_loss.item()
                classification_loss_epoch += classification_loss.item()

            # Calculate average losses for the epoch
            avg_KLD_loss_A = KLD_loss_A_epoch / len(train_loader_A)
            avg_KLD_loss_B = KLD_loss_B_epoch / len(train_loader_B)
            avg_OT_loss = OT_loss_epoch / total_iterations
            avg_classification_loss = classification_loss_epoch / total_iterations

            # Append training loss values to history
            history['KLD_train_loss_A'].append(avg_KLD_loss_A)
            history['KLD_train_loss_B'].append(avg_KLD_loss_B)
            history['OT_train_loss'].append(avg_OT_loss)
            history['classification_train_loss'].append(avg_classification_loss)

            print(f"Avg KLD_A (train) Loss: {avg_KLD_loss_A:.4f}")
            print(f"Avg KLD_B (train) Loss: {avg_KLD_loss_B:.4f}")
            print(f"Avg OT (train) Loss: {avg_OT_loss:.4f}")
            print(f"Avg Classification (train) Loss: {avg_classification_loss:.4f}")

            # Validation phase
            model.eval()
            val_KLD_loss_A, val_KLD_loss_B, val_OT_loss, val_classification_loss, predicted_values, actual_values, accuracy = evaluate_holdout(model, val_loader_A, val_loader_B, nn.BCEWithLogitsLoss(), device)

            # Print or use val_classification_loss in your logging or comparison
            print(f"KLD_A (val) Loss: {val_KLD_loss_A:.4f}")
            print(f"KLD_B (val) Loss: {val_KLD_loss_B:.4f}")
            print(f"OT (val) Loss: {val_OT_loss:.4f}")
            print(f"Classification (val) Loss: {val_classification_loss:.4f}")

            # Log AUROC and AUPRC for the validation
            predicted_probs = torch.sigmoid(predicted_values).cpu().numpy()
            actual_values = actual_values.cpu().numpy()

            auc_score = roc_auc_score(actual_values, predicted_probs)
            print(f"AUROC (from roc_auc_score): {auc_score:.4f}")
            holdout_history['AUROC_val'].append(auc_score)

            auprc_score = average_precision_score(actual_values, predicted_probs)
            print(f"AUPRC (from average_precision_score): {auprc_score:.4f}")
            holdout_history['AUPRC_val'].append(auprc_score)

            # Save the best model if this epoch's validation loss is the lowest
            if val_classification_loss < best_val_loss:
                best_val_loss = val_classification_loss
                best_model_state_fold = model.state_dict()
                best_predicted_values = predicted_values.cpu().numpy()
                best_actual_values = actual_values.cpu().numpy()

            # Early stopping check
            if early_stopper(val_classification_loss):
                print(f"Early stopping at epoch {epoch}")
                break

            scheduler.step(val_classification_loss)

        # Load the best model state from this fold
        model.load_state_dict(best_model_state_fold)

    # Return the best model, history, and the best predicted and actual values
    model.load_state_dict(best_model_state_fold)  # Ensure the best model is returned
    return model, history, holdout_history, best_predicted_values, best_actual_values


def train_model_continuous(model, *datasets, batch_size, learning_rate, m_type, epochs, save_path, splits, device, **kwargs):
    # Prepare DataLoader for batch processing
    train_data = torch.utils.data.TensorDataset(*[torch.tensor(data.values, dtype=torch.float) for data_tuple in datasets for data in data_tuple])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()  # For regression, using MSE loss

    model.train()

    history = {  # Store training history
        'MSE_train_loss': [],
        'MSE_val_loss': [],
        'fold_num': [],
        'epoch_num': [],
    }

    holdout_history = {  # Store holdout evaluation history
        'MSE_eval_loss': [],
        'fold_num': [],
    }

    best_model_state_fold = None  # To store the best model state
    best_val_loss = float('inf')  # Initialize best validation loss
    best_predicted_values = None  # To store predicted values from the best model
    best_actual_values = None  # To store actual values from the best model

    for fold, (pair_A, pair_B) in enumerate(zip(splits.split(datasets[0]), splits.split(datasets[1]))):  # Cross-validation loop
        print(f"Fold {fold + 1}")

        (train_idx_A, val_idx_A) = pair_A
        (train_idx_B, val_idx_B) = pair_B

        # Split the data
        data_A_train, data_A_val = datasets[0][train_idx_A], datasets[0][val_idx_A]
        data_B_train, data_B_val = datasets[1][train_idx_B], datasets[1][val_idx_B]

        train_loader_A = DataLoader(torch.utils.data.TensorDataset(data_A_train, data_A_val), batch_size=batch_size, shuffle=True)
        val_loader_A = DataLoader(torch.utils.data.TensorDataset(data_A_val, data_A_val), batch_size=batch_size, shuffle=False)
        train_loader_B = DataLoader(torch.utils.data.TensorDataset(data_B_train, data_B_val), batch_size=batch_size, shuffle=True)
        val_loader_B = DataLoader(torch.utils.data.TensorDataset(data_B_val, data_B_val), batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, and best weights
        input_dims = [data_A.shape[1], data_B.shape[1]]
        model = Model(*input_dims, dim=kwargs['dim'], dropout=kwargs['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=kwargs['decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=kwargs['learning_gamma'], patience=kwargs['earlystop_patience'], verbose=True)

        best_val_loss = float('inf')
        best_model_state_fold = None

        early_stopper = EarlyStopper(patience=kwargs['earlystop_patience'], min_delta=kwargs['delta'])

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')

            # Training phase
            model.train()

            MSE_loss_epoch = 0.0

            total_iterations = (len(train_loader_A) + len(train_loader_B)) // 2  # Calculate total iterations
            for idx, ((data_A_train, labels_A_train), (data_B_train, labels_B_train)) in tqdm(enumerate(zip(train_loader_A, train_loader_B)), total=total_iterations, desc='Training'):

                data_A_train = data_A_train.to(device)
                data_B_train = data_B_train.to(device)
                labels_A_train = labels_A_train.to(device)
                labels_B_train = labels_B_train.to(device)

                optimizer.zero_grad()

                nets, logits = model(data_A_train, data_B_train)
                (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets

                logits_A, logits_B = logits

                # Late fusion
                logits = torch.cat((logits_A, logits_B), dim=0)
                labels = torch.cat((labels_A_train, labels_B_train), dim=0)

                logits = logits.float()
                labels = labels.float()

                MSE_loss = criterion(logits, labels)

                # Compute the weighted loss using compute_weighted_loss function
                loss = MSE_loss

                loss.backward()
                optimizer.step()

                # Accumulate losses for the epoch
                MSE_loss_epoch += MSE_loss.item()

            # Calculate average losses for the epoch
            avg_MSE_loss = MSE_loss_epoch / total_iterations

            # Append training loss values to history
            history['MSE_train_loss'].append(avg_MSE_loss)

            # Print losses after each epoch during training
            print(f"Avg MSE (train) Loss: {avg_MSE_loss:.4f}")

            # Validation phase
            model.eval()
            val_MSE_loss, predicted_values, actual_values = evaluate_holdout(model, val_loader_A, val_loader_B, nn.MSELoss(), device)

            # Print or use val_MSE_loss in your logging or comparison
            print(f"MSE (val) Loss: {val_MSE_loss:.4f}")

            # Append validation loss values to history
            history['MSE_val_loss'].append(val_MSE_loss)

            # Append fold and epoch numbers to history
            history['fold_num'].append(fold + 1)
            history['epoch_num'].append(epoch + 1)

            # Update best model state if the current model is better
            if val_MSE_loss < best_val_loss:
                best_val_loss = val_MSE_loss
                best_model_state_fold = model.state_dict()
                best_predicted_values = predicted_values
                best_actual_values = actual_values
                print("Updated best validation loss")

            # Check early stopping condition
            if early_stopper(val_MSE_loss):
                print(f'Early stopping at epoch {epoch} with validation loss {val_MSE_loss:.4f}')
                break

            # Early stopping and scheduler step
            scheduler.step(val_MSE_loss)

        # Load the best model state for holdout evaluation
        model.load_state_dict(best_model_state_fold)

        # Evaluate performance on holdout set
        with torch.no_grad():
            holdout_MSE_loss, predicted_values, actual_values = evaluate_holdout(model, holdout_loader_A, holdout_loader_B, nn.MSELoss(), device)

            # Save holdout evaluation losses to history
            holdout_history['MSE_eval_loss'].append(holdout_MSE_loss)
            holdout_history['fold_num'].append(fold + 1)

            # Print holdout evaluation losses after each epoch
            print(f"MSE (eval) Loss: {holdout_MSE_loss:.4f}")

            all_models.append(model.state_dict())

    # Return the model, history, holdout_history, best_predicted_values, and best_actual_values
    return model, history, holdout_history, best_predicted_values, best_actual_values

