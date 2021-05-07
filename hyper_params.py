class HyperParams():
    def __init__(self):
        self.observation_window_hours = 24
        self.antibiotics_name = ['CEFTAZIDIME']
        self.bacteria_ids = [
            80004,  # KLEBSIELLA PNEUMONIAE
            80026,  # PSEUDOMONAS AERUGINOSA
            80005,  # KLEBSIELLA OXYTOCA
            80017,  # PROTEUS MIRABILIS
            80040,  # NEISSERIA GONORRHOEAE
            80008, # ENTEROBACTER CLOACAE
            80007, # ENTEROBACTER AEROGENES
            80002  #ESCHERICHIA COLI
        ]
        self.negative_to_positive_ratio = 3.0
        self.test_set_fraction = 0.10
        self.validation_set_fraction = 0.10
        self.train_set_fraction = 0.80
        self.random_state = 11
        self.num_of_bins_for_numerics = 6
        self.encoder_training_epochs = 170