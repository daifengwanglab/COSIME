class Config:
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    learning_gamma = 0.99
    KLD_A_weight = 0.02
    KLD_B_weight = 0.02
    OT_weight = 0.02
    CL_weight = 0.9
    dropout = 0.5
    dim = 100
    earlystop_patience = 40
    delta = 0.001
    decay = 0.001
    epochs = 100

    # These paths will be filled via command line
    input_data_1 = None  # Set by command line
    input_data_2 = None  # Set by command line
    save_path = None     # Set by command line
    log_path = None      # Set by command line
    type = "binary"      # default "binary", change to "continuous" when needed
    fusion = "early"     # default "early", change to "late" when needed
