# model_loader.py

from models import ModelBinaryEarly, ModelBinaryLate, ModelContinuousEarly, ModelContinuousLate

def load_model(config):
    """
    This function loads the model based on the provided config.

    Args:
        config (dict): A dictionary containing the model configuration parameters.

    Returns:
        model (nn.Module): The model to be trained.
    """
    if config['type'] == 'binary' and config['fusion'] == 'early':
        return ModelBinaryEarly()
    elif config['type'] == 'binary' and config['fusion'] == 'late':
        return ModelBinaryLate()
    elif config['type'] == 'continuous' and config['fusion'] == 'early':
        return ModelContinuousEarly()
    elif config['type'] == 'continuous' and config['fusion'] == 'late':
        return ModelContinuousLate()
    else:
        raise ValueError("Invalid model configuration")
