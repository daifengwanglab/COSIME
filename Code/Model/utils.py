import logging

# Early Stopper class for early stopping during training
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# Function for logging the results during training
def log_results(epoch, loss, epochs, log_path, **kwargs):
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")
