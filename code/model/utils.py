# utils.py

import logging

def log_results(epoch, loss, config):
    logging.basicConfig(filename=config['log_path'], level=logging.INFO)
    logging.info(f"Epoch [{epoch}/{config['epochs']}], Loss: {loss.item()}")
