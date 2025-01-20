import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from loss import compute_weighted_loss, KL_divergence, LOT
from utils import EarlyStopper
from torch.utils.data import DataLoader, Subset
from data_loader_revised import load_and_prepare_data

def train_model_binary(model, data1_path, data2_path, batch_size, learning_rate, m_type, epochs, save_path, splits, device, **kwargs):
    """
    Train the binary classification model using data from two CSV files.

    :param model: The model to be trained.
    :param data1_path: Path to the first CSV data file.
    :param data2_path: Path to the second CSV data file.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param m_type: Type of model ('binary' or 'continuous').
    :param epochs: Number of epochs for training.
    :param save_path: Path to save the model after training.
    :param splits: Cross-validation splits.
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param kwargs: Additional keyword arguments (e.g., for early stopping, learning rate scheduler).
    """

    # Load and prepare data using the load_and_prepare_data function
    train_loader_A, val_loader_A, holdout_loader_A, train_loader_B, val_loader_B, holdout_loader_B = load_and_prepare_data(
        batch_size=batch_size,
        data1_path=data1_path,
        data2_path=data2_path,
        task_type='binary',  # explicitly specifying binary task
        **kwargs
    )

    # Initialize optimizer, loss function, and training history
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary classification loss

    model.train()

    history = {
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

    best_model_state = None
    best_val_loss = float('inf')
    best_predicted_values = None
    best_actual_values = None

    # Loop over cross-validation splits
    for fold, (pair_A, pair_B) in enumerate(zip(splits.split(train_loader_A.dataset), splits.split(train_loader_B.dataset))):
        print(f"Fold {fold + 1}")

        # Cross-validation: Split training and validation data
        (train_idx_A, val_idx_A) = pair_A
        (train_idx_B, val_idx_B) = pair_B

        # Get the datasets for training and validation
        train_loader_A_fold = DataLoader(Subset(train_loader_A.dataset, train_idx_A), batch_size=batch_size, shuffle=True)
        val_loader_A_fold = DataLoader(Subset(val_loader_A.dataset, val_idx_A), batch_size=batch_size)
        
        train_loader_B_fold = DataLoader(Subset(train_loader_B.dataset, train_idx_B), batch_size=batch_size, shuffle=True)
        val_loader_B_fold = DataLoader(Subset(val_loader_B.dataset, val_idx_B), batch_size=batch_size)

        # Initialize early stopper
        early_stopper = EarlyStopper(patience=kwargs.get('earlystop_patience', 10), min_delta=kwargs.get('delta', 0.001))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")

            model.train()
            KLD_loss_A_epoch = 0.0
            KLD_loss_B_epoch = 0.0
            OT_loss_epoch = 0.0
            classification_loss_epoch = 0.0

            # Training loop
            for (data_A, labels_A), (data_B, labels_B) in tqdm(zip(train_loader_A_fold, train_loader_B_fold), total=len(train_loader_A_fold)):
                data_A, labels_A = data_A.to(device), labels_A.to(device)
                data_B, labels_B = data_B.to(device), labels_B.to(device)

                optimizer.zero_grad()

                # Forward pass
                nets, logits = model(data_A, data_B)
                (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets
                logits_A, logits_B = logits

                # Compute loss components
                KLD_loss_A = KL_divergence(mu_A, logsigma_A)
                KLD_loss_B = KL_divergence(mu_B, logsigma_B)
                OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

                # Combine logits and labels for loss calculation
                logits = torch.cat((logits_A, logits_B), dim=0)
                labels = torch.cat((labels_A, labels_B), dim=0)

                # Classification loss
                classification_loss = criterion(logits.squeeze(), labels.squeeze())

                # Compute the total weighted loss
                loss = compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, classification_loss,
                                             kwargs['kld_1_weight'], kwargs['kld_2_weight'], kwargs['ot_weight'], kwargs['cl_weight'])

                loss.backward()
                optimizer.step()

                # Accumulate loss for reporting
                KLD_loss_A_epoch += KLD_loss_A.item()
                KLD_loss_B_epoch += KLD_loss_B.item()
                OT_loss_epoch += OT_loss.item()
                classification_loss_epoch += classification_loss.item()

            # Calculate average losses for the epoch
            avg_KLD_loss_A = KLD_loss_A_epoch / len(train_loader_A_fold)
            avg_KLD_loss_B = KLD_loss_B_epoch / len(train_loader_B_fold)
            avg_OT_loss = OT_loss_epoch / len(train_loader_A_fold)
            avg_classification_loss = classification_loss_epoch / len(train_loader_A_fold)

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
            val_KLD_loss_A, val_KLD_loss_B, val_OT_loss, val_classification_loss, predicted_values, actual_values, accuracy = evaluate_holdout(
                model, val_loader_A_fold, val_loader_B_fold, nn.BCEWithLogitsLoss(), device
            )

            print(f"KLD_A (val) Loss: {val_KLD_loss_A:.4f}")
            print(f"KLD_B (val) Loss: {val_KLD_loss_B:.4f}")
            print(f"OT (val) Loss: {val_OT_loss:.4f}")
            print(f"Classification (val) Loss: {val_classification_loss:.4f}")

            # Log AUROC and AUPRC for the validation
            predicted_probs = torch.sigmoid(predicted_values).cpu().numpy()
            actual_values = actual_values.cpu().numpy()

            auc_score = roc_auc_score(actual_values, predicted_probs)
            print(f"AUROC (from roc_auc_score): {auc_score:.4f}")
            history['AUROC_val'].append(auc_score)

            auprc_score = average_precision_score(actual_values, predicted_probs)
            print(f"AUPRC (from average_precision_score): {auprc_score:.4f}")
            history['AUPRC_val'].append(auprc_score)

            # Save the best model if this epoch's validation loss is the lowest
            if val_classification_loss < best_val_loss:
                best_val_loss = val_classification_loss
                best_model_state = model.state_dict()
                best_predicted_values = predicted_values.cpu().numpy()
                best_actual_values = actual_values.cpu().numpy()

            # Early stopping check
            if early_stopper(val_classification_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        # Load the best model state from this fold
        model.load_state_dict(best_model_state)

    # Return the best model and history
    return model, history, best_predicted_values, best_actual_values

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from loss import compute_weighted_loss, KL_divergence, LOT
from utils import EarlyStopper
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error
from data_loader_revised import load_and_prepare_data

def train_model_continuous(model, data1_path, data2_path, batch_size, learning_rate, m_type, epochs, save_path, splits, device, **kwargs):
    """
    Train the continuous regression model using data from two CSV files.
    
    :param model: The model to be trained.
    :param data1_path: Path to the first CSV data file.
    :param data2_path: Path to the second CSV data file.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param m_type: Type of model ('binary' or 'continuous').
    :param epochs: Number of epochs for training.
    :param save_path: Path to save the model after training.
    :param splits: Cross-validation splits.
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param kwargs: Additional keyword arguments (e.g., for early stopping, learning rate scheduler).
    """

    # Load and prepare data using the load_and_prepare_data function for continuous task
    train_loader_A, val_loader_A, holdout_loader_A, train_loader_B, val_loader_B, holdout_loader_B = load_and_prepare_data(
        batch_size=batch_size,
        data1_path=data1_path,
        data2_path=data2_path,
        task_type='continuous',  # explicitly specifying continuous task
        **kwargs
    )

    # Initialize optimizer, loss function, and training history
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # For continuous regression tasks

    model.train()

    history = {
        'KLD_train_loss_A': [],
        'KLD_train_loss_B': [],
        'OT_train_loss': [],
        'regression_train_loss': [],
        'KLD_val_loss_A': [],
        'KLD_val_loss_B': [],
        'OT_val_loss': [],
        'regression_val_loss': [],
        'fold_num': [],
        'epoch_num': [],
    }

    best_model_state = None
    best_val_loss = float('inf')
    best_predicted_values = None
    best_actual_values = None

    # Loop over cross-validation splits
    for fold, (pair_A, pair_B) in enumerate(zip(splits.split(train_loader_A.dataset), splits.split(train_loader_B.dataset))):
        print(f"Fold {fold + 1}")

        # Cross-validation: Split training and validation data
        (train_idx_A, val_idx_A) = pair_A
        (train_idx_B, val_idx_B) = pair_B

        # Get the datasets for training and validation
        train_loader_A_fold = DataLoader(Subset(train_loader_A.dataset, train_idx_A), batch_size=batch_size, shuffle=True)
        val_loader_A_fold = DataLoader(Subset(val_loader_A.dataset, val_idx_A), batch_size=batch_size)
        
        train_loader_B_fold = DataLoader(Subset(train_loader_B.dataset, train_idx_B), batch_size=batch_size, shuffle=True)
        val_loader_B_fold = DataLoader(Subset(val_loader_B.dataset, val_idx_B), batch_size=batch_size)

        # Initialize early stopper
        early_stopper = EarlyStopper(patience=kwargs.get('earlystop_patience', 10), min_delta=kwargs.get('delta', 0.001))

        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=kwargs.get('learning_gamma', 0.1),
                                                               patience=kwargs.get('earlystop_patience', 10), verbose=True)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")

            model.train()
            KLD_loss_A_epoch = 0.0
            KLD_loss_B_epoch = 0.0
            OT_loss_epoch = 0.0
            regression_loss_epoch = 0.0

            total_iterations = (len(train_loader_A) + len(train_loader_B)) // 2  # Calculate total iterations
            for idx, ((data_A_train, labels_A_train), (data_B_train, labels_B_train)) in tqdm(enumerate(zip(train_loader_A_fold, train_loader_B_fold)), 
                                                                                        total=total_iterations, desc='Training'):
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

                KLD_loss_A = KL_divergence(mu_A, logsigma_A)
                KLD_loss_B = KL_divergence(mu_B, logsigma_B)

                regression_loss = criterion(logits, labels)

                OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

                # Compute the weighted loss using compute_weighted_loss function
                loss = compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, regression_loss, 
                                             kwargs['KLD_A_weight'], kwargs['KLD_B_weight'], kwargs['OT_weight'], kwargs['regression_weight'])

                loss.backward()
                optimizer.step()

                # Accumulate losses for the epoch
                KLD_loss_A_epoch += KLD_loss_A.item()
                KLD_loss_B_epoch += KLD_loss_B.item()
                OT_loss_epoch += OT_loss.item()
                regression_loss_epoch += regression_loss.item()

            # Calculate average losses for the epoch
            avg_KLD_loss_A = KLD_loss_A_epoch / len(train_loader_A_fold)
            avg_KLD_loss_B = KLD_loss_B_epoch / len(train_loader_B_fold)
            avg_OT_loss = OT_loss_epoch / total_iterations
            avg_regression_loss = regression_loss_epoch / total_iterations

            # Append training loss values to history
            history['KLD_train_loss_A'].append(avg_KLD_loss_A)
            history['KLD_train_loss_B'].append(avg_KLD_loss_B)
            history['OT_train_loss'].append(avg_OT_loss)
            history['regression_train_loss'].append(avg_regression_loss)

            # Print losses after each epoch during training
            print(f"Avg KLD_A (train) Loss: {avg_KLD_loss_A:.4f}")
            print(f"Avg KLD_B (train) Loss: {avg_KLD_loss_B:.4f}")
            print(f"Avg OT (train) Loss: {avg_OT_loss:.4f}")
            print(f"Avg Regression (train) Loss: {avg_regression_loss:.4f}")

            # Validation phase
            model.eval()
            val_KLD_loss_A, val_KLD_loss_B, val_OT_loss, val_regression_loss, predicted_values, actual_values = evaluate_holdout(
                model, val_loader_A_fold, val_loader_B_fold, nn.MSELoss(), device
            )

            # Print or use val_regression_loss in your logging or comparison
            print(f"KLD_A (val) Loss: {val_KLD_loss_A:.4f}")
            print(f"KLD_B (val) Loss: {val_KLD_loss_B:.4f}")
            print(f"OT (val) Loss: {val_OT_loss:.4f}")
            print(f"Regression (val) Loss: {val_regression_loss:.4f}")

            # Append validation loss values to history
            history['KLD_val_loss_A'].append(val_KLD_loss_A)
            history['KLD_val_loss_B'].append(val_KLD_loss_B)
            history['OT_val_loss'].append(val_OT_loss)
            history['regression_val_loss'].append(val_regression_loss)

            # Append fold and epoch numbers to history
            history['fold_num'].append(fold + 1)
            history['epoch_num'].append(epoch + 1)

            # Update best model state if the current model is better
            if val_regression_loss < best_val_loss:
                best_val_loss = val_regression_loss
                best_model_state_fold = model.state_dict()
                print("Updated best validation loss")

            else:
                print(f"No improvement: val_regression_loss={val_regression_loss:.4f}, best_val_loss={best_val_loss:.4f}")

            # Check early stopping condition
            if early_stopper(val_regression_loss):
                print(f'Early stopping at epoch {epoch} with validation loss {val_regression_loss:.4f}')
                break

            # Early stopping and scheduler step
            print(f"Scheduler step with val_regression_loss: {val_regression_loss:.4f}")
            scheduler.step(val_regression_loss)

            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

        # Load the best model state for holdout evaluation
        model.load_state_dict(best_model_state_fold)

        # Evaluate performance on holdout set
        with torch.no_grad():
            holdout_KLD_loss_A, holdout_KLD_loss_B, holdout_OT_loss, holdout_regression_loss, predicted_values, actual_values = evaluate_holdout(
                model, holdout_loader_A, holdout_loader_B, nn.MSELoss(), device
            )

            predicted_values = torch.tensor(predicted_values)
            actual_values = torch.tensor(actual_values)

            # Save holdout evaluation losses to history
            holdout_history = {
                'KLD_eval_loss_A': holdout_KLD_loss_A,
                'KLD_eval_loss_B': holdout_KLD_loss_B,
                'OT_eval_loss': holdout_OT_loss,
                'regression_eval_loss': holdout_regression_loss
            }

            print(f"KLD_A (eval) Loss: {holdout_KLD_loss_A:.4f}")
            print(f"KLD_B (eval) Loss: {holdout_KLD_loss_B:.4f}")
            print(f"OT (eval) Loss: {holdout_OT_loss:.4f}")
            print(f"Regression (eval) Loss: {holdout_regression_loss:.4f}")

    return model, history

