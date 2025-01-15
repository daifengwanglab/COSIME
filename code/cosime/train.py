import torch
from torch.utils.data import DataLoader

from loss import compute_weighted_loss, KL_divergence, LOT
from utils import log_results

def train_model(model, *datasets, batch_size, learning_rate, m_type, epochs, save_path, **kwargs):
    # Prepare DataLoader for batch processing
    train_data = torch.utils.data.TensorDataset(*[torch.tensor(data.values, dtype=torch.float) for data_tuple in datasets for data in data_tuple])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss() if m_type == 'binary' else torch.nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            # Load data
            X1, y1, X2, y2 = batch
            optimizer.zero_grad()

            # Checks
            assert len(batch) == 4, 'Must have two datasets as input'
            assert (y1 == y2).all(), 'Datasets not aligned'

            # Forward pass
            net_logits, nn_logits = model(X1, X2)
            (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = net_logits

            # Late handling
            if kwargs['fusion'] == 'late':
                logits = torch.cat(nn_logits, dim=0)
                labels = torch.cat((y1, y2), dim=0)
            elif kwargs['fusion'] == 'early':
                logits, labels = nn_logits, y1
            else: raise ValueError(f'Fusion {kwargs["fusion"]} unsupported.')

            # KLD Loss
            KLD_loss_A = KL_divergence(mu_A, logsigma_A)
            KLD_loss_B = KL_divergence(mu_B, logsigma_B)

            # OT Loss
            OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

            # Classification loss
            classification_loss = criterion(logits.squeeze(), labels)

            # Compute the weighted loss using compute_weighted_loss function
            loss = compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, classification_loss, kwargs['kld_1_weight'], kwargs['kld_2_weight'], kwargs['ot_weight'], kwargs['cl_weight'])

            # Step
            loss.backward()
            optimizer.step()

        # Log results
        log_results(epoch, loss, epochs, **kwargs)

    # Save the model
    torch.save(model.state_dict(), save_path)
