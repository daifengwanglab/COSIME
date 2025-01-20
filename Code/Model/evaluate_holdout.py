import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
import torch

from loss import compute_weighted_loss, KL_divergence, LOT


def evaluate_holdout_binary(model, data_loader_A, data_loader_B, criterion, fusion, device):
    model.eval()

    total_KLD_loss_A = 0
    total_KLD_loss_B = 0
    total_OT_loss = 0
    total_classification_loss = 0
    num_batches = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_A, batch_B in zip(data_loader_A, data_loader_B):

            data_A, labels_A = batch_A
            data_B, labels_B = batch_B

            data_A = data_A.to(device)
            data_B = data_B.to(device)
            labels_A = labels_A.to(device)
            labels_B = labels_B.to(device)

            data_A_tensors = data_A.clone().detach().requires_grad_(True)
            data_B_tensors = data_B.clone().detach().requires_grad_(True)

            nets, logits = model(data_A_tensors, data_B_tensors)
            (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets

            KLD_loss_A = KL_divergence(mu_A, logsigma_A)
            KLD_loss_B = KL_divergence(mu_B, logsigma_B)

            # Late fusion
            if fusion == 'late':
                logits_A, logits_B = logits
                logits = torch.cat((logits_A, logits_B), dim=0)
                labels = torch.cat((labels_A, labels_B), dim=0)
            else: labels = labels_A.float()

            OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

            # Check if logits is a list
            if isinstance(logits, list):
                # Convert list of tensors to a single tensor
                logits = torch.cat(logits, dim=1)

            # Unpack logits
            classification_logits = logits.squeeze()

            # Reshape labels to match the shape of classification_logits
            labels = labels.float().view(-1, 1)

            classification_loss = criterion(classification_logits, labels.squeeze())

            # Accumulate losses
            total_KLD_loss_A += KLD_loss_A.item()
            total_KLD_loss_B += KLD_loss_B.item()
            total_OT_loss += OT_loss.item()
            total_classification_loss += classification_loss.item()
            num_batches += 1

            # Store logits and labels for accuracy calculation
            all_logits.append(classification_logits)
            all_labels.append(labels)

    avg_KLD_loss_A = total_KLD_loss_A / num_batches
    avg_KLD_loss_B = total_KLD_loss_B / num_batches
    avg_OT_loss = total_OT_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches

    # Concatenate all logits and labels
    all_logits = torch.cat(all_logits, dim=0).squeeze()
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    # Calculate prediction probabilities
    predicted_probs = torch.sigmoid(all_logits)

    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels.cpu().numpy(), predicted_probs.cpu().numpy())
    
    # Calculate F1 scores for each threshold
    f1_scores = [f1_score(all_labels.cpu().numpy(), predicted_probs.cpu().numpy() >= t) for t in thresholds]

    # Find the threshold with the maximum F1 score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    # Debugging threshold choice
    # print(f"Thresholds and their F1 scores: {list(zip(thresholds, f1_scores))}")
    print(f"Best threshold: {best_threshold}, Best F1 score: {best_f1_score}")

    # Make predictions using the best threshold
    predicted_classes = (predicted_probs >= best_threshold).to(torch.int)

    # Calculate final accuracy
    correct_predictions = (predicted_classes == all_labels).sum().item()
    accuracy = correct_predictions / len(all_labels)

    # Debugging threshold choice
    # print(f"Thresholds and their F1 scores: {list(zip(thresholds, f1_scores))}")
    print(f"Best threshold: {best_threshold}, Best F1 score: {best_f1_score}")

    return avg_KLD_loss_A, avg_KLD_loss_B, avg_OT_loss, avg_classification_loss, all_logits, all_labels, accuracy



def evaluate_holdout_continuous(model, data_loader_A, data_loader_B, criterion, fusion, device):
    model.eval()

    total_KLD_loss_A = 0
    total_KLD_loss_B = 0
    total_OT_loss = 0
    total_classification_loss = 0
    num_batches = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_A, batch_B in zip(data_loader_A, data_loader_B):

            data_A, labels_A = batch_A
            data_B, labels_B = batch_B

            data_A = data_A.to(device)
            data_B = data_B.to(device)
            labels_A = labels_A.to(device)
            labels_B = labels_B.to(device)

            data_A = torch.tensor(data_A, dtype=torch.float32)
            data_B = torch.tensor(data_B, dtype=torch.float32)

            data_A_tensors = data_A.clone().detach().requires_grad_(True)
            data_B_tensors = data_B.clone().detach().requires_grad_(True)

            nets, logits = model(data_A_tensors, data_B_tensors)
            (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets

            KLD_loss_A = KL_divergence(mu_A, logsigma_A)
            KLD_loss_B = KL_divergence(mu_B, logsigma_B)
            OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

            # Late fusion
            if fusion == 'late':
                logits_A, logits_B = logits
                logits = torch.cat((logits_A, logits_B), dim=0)
                labels = torch.cat((labels_A, labels_B), dim=0)
            else: labels = labels_A.float()

            # Check if logits is a list
            if isinstance(logits, list):
                # Convert list of tensors to a single tensor
                logits = torch.cat(logits, dim=1)

            # Unpack logits
            classification_logits = logits.squeeze()

            # Reshape labels to match the shape of classification_logits
            labels = labels.float().view(-1, 1)

            classification_loss = criterion(classification_logits, labels.squeeze())
            
            # Accumulate losses
            total_KLD_loss_A += KLD_loss_A.item()
            total_KLD_loss_B += KLD_loss_B.item()
            total_OT_loss += OT_loss.item()
            total_classification_loss += classification_loss.item()
            num_batches += 1

            # Store logits and labels for accuracy
            all_logits.append(classification_logits)
            all_labels.append(labels)

    avg_KLD_loss_A = total_KLD_loss_A / num_batches
    avg_KLD_loss_B = total_KLD_loss_B / num_batches
    avg_OT_loss = total_OT_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches

    all_logits = torch.cat(all_logits, dim=0).squeeze()
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    return avg_KLD_loss_A, avg_KLD_loss_B, avg_OT_loss, avg_classification_loss, all_logits, all_labels

