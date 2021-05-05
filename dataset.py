import config
from hyper_params import HyperParams

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


def build_autoencoded_dataframe(numpy_output_file='autoencoded_fulldata.npy', dataframe_output_file='df_autoencoded_cohort'):
    pass
    # params = HyperParams()
    #
    # # 1. build all features dataset, for all 54k admissions
    # df_final_dataset = featues_datasets_all_patients.run(params, binning_numerics=True, create_patients_list_view=True, create_lab_events=True)
    # print(f"Created full features dataset: {df_final_dataset.shape}")
    #
    # # 2. Train the AutoEncoder
    # encoder_training_epochs = params.encoder_training_epochs
    # dataset = TheDataSet(datafile=dataset_file, pad_to_360=False)
    # print(f"dataset length = {len(dataset)} num features = {dataset.num_features()}")
    # model = Autoencoder(num_features=dataset.num_features())
    # print(model)
    # max_epochs = encoder_training_epochs
    # outputs, losses = train(model, dataset=dataset, num_epochs=max_epochs, batch_size=512, learning_rate=1e-3)
    #
    # # 2. build a labeled cohort
    # np_datafile = config.DATA_DIR + '/' + numpy_output_file
    # df_cohort = build_cohort_dataset.build_cohort(params, df_final_dataset, np_datafile)
    # print(f"Created cohort dataset: {df_final_dataset.shape}")
    #
    # io.write_dataframe(df_cohort, dataframe_output_file)
