# train.py

import torch
from torch.utils.data import DataLoader
from utils import log_results

def train_model(model, data_1, data_2, config):
    # Prepare DataLoader for batch processing
    train_data = torch.utils.data.TensorDataset(torch.tensor(data_1.values), torch.tensor(data_2.values))
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss() if config['type'] == 'binary' else torch.nn.MSELoss()

    model.train()

    for epoch in range(config['epochs']):
        for batch in train_loader:
            inputs_1, inputs_2 = batch
            optimizer.zero_grad()

            # Forward pass
            outputs, logits = model(inputs_1, inputs_2)
            
            # Compute the loss
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        # Log results
        log_results(epoch, loss, config)

    # Save the model
    torch.save(model.state_dict(), config['save_path'])
