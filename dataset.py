import torch
from torch.utils.data import Dataset
import numpy as np

import config
from hyper_params import HyperParams
from embeddings.dataloader import TheDataSet

from dataproc import io, featues_datasets_all_patients, build_cohort_dataset


def build_normal_dataframe(numpy_output_file='fulldata.npy', dataframe_output_file='df_cohort'):
    params = HyperParams()

    # 1. build all features dataset, for all 54k admissions
    df_final_dataset = featues_datasets_all_patients.run(params, binning_numerics=False, create_patients_list_view=True, create_lab_events=True)
    print(f"Created full features dataset: {df_final_dataset.shape}")

    # 2. build a labeled cohort
    np_datafile = config.DATA_DIR + '/' + numpy_output_file
    df_cohort = build_cohort_dataset.build_cohort(params, df_final_dataset, np_datafile)
    print(f"Created cohort dataset: {df_final_dataset.shape}")

    io.write_dataframe(df_cohort, dataframe_output_file)
    return df_cohort


def build_autoencoded_data_matrix(numpy_output_file='autoencoded_fulldata.npy', params = HyperParams()):
    
    # 1. build all features dataset, for all 54k admissions
    df_final_dataset_binned = featues_datasets_all_patients.run(params, binning_numerics=True, create_patients_list_view=True, create_lab_events=True)
    print(f"Created full features dataset: {df_final_dataset_binned.shape}")
    io.write_dataframe(df_final_dataset_binned, 'df_final_dataset_binned')
    df_final_dataset_binned = io.load_dataframe('df_final_dataset_binned')
    
    
    # write AE training data to numpy file
    ae_training_datafile_name = 'autoencoder_training_data.npy'
    np_training_datafile = config.DATA_DIR + '/' + ae_training_datafile_name
    print(f"Writing AutoEncoder training data to {np_training_datafile}")
    featues_datasets_all_patients.save_auto_encoder_training_data(
        df_final_dataset_binned, 
        target_datafile = np_training_datafile
    )
    # 2. Train the AutoEncoder
    encoder_training_epochs = params.encoder_training_epochs
    dataset = TheDataSet(datafile=np_training_datafile)
    print(f"dataset length = {len(dataset)} num features = {dataset.num_features()}")
    from embeddings.autoencoder import Autoencoder
    from embeddings.train import train, plot_loss
    model = Autoencoder(num_features=dataset.num_features())
    print(model)
    max_epochs = encoder_training_epochs
    outputs, losses = train(model, dataset=dataset, num_epochs=max_epochs, batch_size=512, learning_rate=1e-3, denoising=True, denoise_p=0.1)
    io.write_serialized_model(model, 'autoencoder')
    print(f"Trained AutoEncoder. Training Data Loss Reached: {losses[-1]} ")
    plot_loss(losses)

    model = io.load_serialized_model('autoencoder')
    
    # 2. build a labeled cohort
    np_cohort_data_file = config.DATA_DIR + '/' + 'raw_cohort_data.npy'
    df_cohort = build_cohort_dataset.build_cohort(params, df_final_dataset_binned, np_cohort_data_file)
    print(f"Created cohort dataset: {df_cohort.shape}")
    
    # 3. Encode the cohort using the trained AutoEncoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cohort_dataset = TheDataSet(datafile=np_cohort_data_file)
    data_loader = torch.utils.data.DataLoader(cohort_dataset, batch_size=1, shuffle=False)
    rows=[]
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        row = model.encoder(X.float())
        row = torch.cat([row.reshape(1,-1),y.reshape(1,-1).float()], dim=1)
        rows.append(row)
    encoded_data = torch.cat(rows, dim=0)
    np_labeled_data = encoded_data.detach().to('cpu').numpy()
    numpy_output_file = config.DATA_DIR + '/' + numpy_output_file
    print(f"Writing cohort matrix to {numpy_output_file}")
    np.save(numpy_output_file, np_labeled_data)
    print(f"Created cohort matrix: {np_labeled_data.shape}")
    return np_labeled_data

