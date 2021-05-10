class HyperParams():
    def __init__(self):
        self.observation_window_hours = 24
        self.encoder_training_epochs = 170
        self.num_of_bins_for_numerics = 6
        self.negative_to_positive_ratio = 3.0
        self.test_set_fraction = 0.30
        self.validation_set_fraction = 0.10
        self.train_set_fraction = 0.80
        self.random_state = 11
