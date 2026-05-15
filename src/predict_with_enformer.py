


import tensorflow as tf
import numpy as np
import sys, h5py

def get_model(model_path):
    """
    Return a tensorflow model

    Parameters:
        model_path: str
            A path to where the tensorflow model exists
    Returns: 
        a tensorflow model
    """
    import tensorflow as tf
    return tf.saved_model.load(model_path).model


enformer_model = get_model('/project2/haky/Data/enformer/raw')
db_file = f"/scratch/beagle3/temi/alphagenome/genes_reference/random_100_genes.reference_onehot.hdf5"

# read in the one hot encodings
with h5py.File(db_file, 'r') as rf:
    one_hot_encodings = rf['reference_onehot']['100_genes'][:, :, :]

prediction_file = f"/scratch/beagle3/temi/alphagenome/genes_reference/random_100_genes.reference_enformer_predictions.hdf5"

input_size = one_hot_encodings.shape

with h5py.File(prediction_file, "w") as f:
    group1 = f.create_group('reference_enformer_cage')
    dset1 = group1.create_dataset("100_genes", (input_size[0], 896, 5313), dtype='f16')
    for i in range(input_size[0]):
        print(f"INFO - Prediction for gene {i} of {input_size[0]}")
        try:
            input_matrix = np.expand_dims(one_hot_encodings[i, :], axis = 0) 
            enf_prediction = enformer_model.predict_on_batch(input_matrix)['human'].numpy().squeeze()
            dset1[i, :, :] = enf_prediction[:, :]
        except:
            print(f"The loop breaks at prediction {i} ;( ")
            continue # enformer_model.predict_on_batch(enf_ohe)['human'].numpy()

print("INFO - Done with predictions!")