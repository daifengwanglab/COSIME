import logging

def log_results(epoch, loss, epochs, log_path, **kwargs):
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")
