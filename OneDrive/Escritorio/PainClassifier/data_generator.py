import os
import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt




class DefineParameters:
    def __init__(self,model, batch_train=1, epochs=1, batch_test=1, fMRI_Modality=None, subjects=None, sessions=None):
        self.rootpath = 'C:/Users/damia/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey' \
                        '/rawdata/'
        self.fMRI_Modality = fMRI_Modality
        self.subjects = subjects
        self.sessions = sessions
        self.batch_train = batch_train
        self.steps_per_epoch = int(np.ceil(620 * self.batch_train))
        self.epochs = epochs
        self.batch_test = batch_test
        self.subjects_names = self.get_subjects_names()
        self.sessions_names = self.get_sessions_names()
        self.filenames, self.label_names = self.get_ID_filenames()
        self.n_classes = int()
        self.auc = float()
        self.precision = float()
        self.recall = float()
        self.f1 = float()
        self.model = model

    def get_subjects_names(self):
        ID_subjects = []
        for sub in self.subjects:
            if 0 < int(sub) < 182:
                subj_ID = 'sub-' + str.zfill(str(sub), 3)
                ID_subjects.append(subj_ID)
        return ID_subjects

    def get_sessions_names(self):
        ID_sessions = []
        for ses in self.sessions:
            if 0 < int(ses) <= 3:
                ses_ID = 'ses-' + str.zfill(str(ses), 2)
                ID_sessions.append(ses_ID)
        return ID_sessions

    def get_ID_filenames(self):
        files = []
        label_files = []
        label_table = pd.read_table(self.rootpath + '/participants.tsv', encoding='utf-8')
        for subj in self.subjects_names:
            for sess in self.sessions_names:
                file = subj + '/' + sess + '/func/' + subj + '_' + sess + '_task-' + self.fMRI_Modality + '_bold.nii'
                # print(file)
                if os.path.exists(self.rootpath + file):
                    files.append(file)
                    label = get_label_from_tsv_file(label_table, subj, sess)
                    label_files.append(label)

        return files, label_files

def train_data_generator(self, train_ix):
    j = 0
    train_feature_pool_filenames = [self.filenames[index] for index in train_ix]
    train_label_pool = [self.label_names[index] for index in train_ix]
    
    #transform to onehot
    onehot_encoder = OneHotEncoder(sparse_output=False)
    train_label_pool = np.array(train_label_pool).reshape(len(train_label_pool), 1)
    train_label_pool = onehot_encoder.fit_transform(train_label_pool)
    
    while True:
        if j * self.batch_train >= len(train_ix):  # This loop is used to run the generator indefinitely.
            j = 0
            # random.shuffle(self.subjects)
        else:
            filenames_chunk = train_feature_pool_filenames[j * self.batch_train:(j + 1) * self.batch_train]
            label_chunk = train_label_pool[j * self.batch_train:(j + 1) * self.batch_train]
            inputs = []
            targets = []
            for idx_filename in range(len(filenames_chunk)):
                temp = nib.load(self.rootpath + filenames_chunk[idx_filename])
                vol = temp.get_fdata()
                vol = np.transpose(vol, (3, 0, 1, 2))
                inputs.append(vol)
                labels = np.ones((vol.shape[0],len(label_chunk[idx_filename])), dtype=int) * label_chunk[idx_filename]
                targets.extend(labels)

            inputs = np.asarray(inputs)
            targets = np.asarray(targets).reshape(vol.shape[0],len(label_chunk[idx_filename]))
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs = np.asarray([data for data in data_ds]).reshape(620, 128, 128, 22)
            yield inputs, targets
            j = j + 1
            
def train_data_generator_short(self, train_ix):
    j = 0
    train_feature_pool_filenames = [self.filenames[index] for index in train_ix]
    train_label_pool = [self.label_names[index] for index in train_ix]
    
    #transform to onehot
    onehot_encoder = OneHotEncoder(sparse_output=False)
    train_label_pool = np.array(train_label_pool).reshape(len(train_label_pool), 1)
    train_label_pool = onehot_encoder.fit_transform(train_label_pool)
    
    
    while True:
        if j * self.batch_train >= len(train_ix):  # This loop is used to run the generator indefinitely.
            j = 0
            # random.shuffle(self.subjects)
        else:
            filenames_chunk = train_feature_pool_filenames[j * self.batch_train:(j + 1) * self.batch_train]
            label_chunk = train_label_pool[j * self.batch_train:(j + 1) * self.batch_train]
            inputs = []
            targets = []
            for idx_filename in range(len(filenames_chunk)):
                temp = nib.load(self.rootpath + filenames_chunk[idx_filename])
                vol = temp.get_fdata()
                vol = np.transpose(vol, (3, 0, 1, 2))
                vol1 = vol[0]
                inputs.append(vol1)
                
                labels = np.ones((1,len(label_chunk[idx_filename])), dtype=int) * label_chunk[idx_filename]
                targets.append(labels)

            inputs = np.asarray(inputs)
            
            targets = np.asarray(targets).reshape(inputs.shape[0],len(label_chunk[idx_filename]))
            
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs = np.asarray([data for data in data_ds]).reshape(first_dim, 128, 128, 22)
            yield inputs, targets
            j = j + 1

def test_data_generator(self, test_ix):
    j = 0
    test_feature_pool_filenames = [self.filenames[index] for index in test_ix]
    
    while True:
        if j * self.batch_train >= len(test_ix):  # This loop is used to run the generator indefinitely.
            j = 0
            # random.shuffle(self.subjects)
        else:
            filenames_chunk = test_feature_pool_filenames[j * self.batch_test:(j + 1) * self.batch_test]
            
            inputs = []
            for idx_filename in range(len(filenames_chunk)):
                temp = nib.load(self.rootpath + filenames_chunk[idx_filename])
                vol = temp.get_fdata()
                vol = np.transpose(vol, (3, 0, 1, 2))
                inputs.append(vol)

            inputs = np.asarray(inputs)
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs = np.asarray([data for data in data_ds]).reshape(first_dim, 128, 128, 22)
            yield inputs
            j = j + 1
            
def test_data_generator_short(self, test_ix):
    j = 0
    test_feature_pool_filenames = [self.filenames[index] for index in test_ix]
    test_label_pool = [self.label_names[index] for index in test_ix]
    
    #transform to onehot
    onehot_encoder = OneHotEncoder(sparse_output=False)
    test_label_pool = np.array(test_label_pool).reshape(len(test_label_pool), 1)
    test_label_pool = onehot_encoder.fit_transform(test_label_pool)
    
    while True:
        if j * self.batch_test >= len(test_ix):  # This loop is used to run the generator indefinitely.
            j = 0
            # random.shuffle(self.subjects)
        else:
            filenames_chunk = test_feature_pool_filenames[j * self.batch_test:(j + 1) * self.batch_test]
            #label_chunk = test_label_pool[j * self.batch_test:(j + 1) * self.batch_test]
            #targets = []      
            inputs = []
            for idx_filename in range(len(filenames_chunk)):
                temp = nib.load(self.rootpath + filenames_chunk[idx_filename])
                vol = temp.get_fdata()
                vol = np.transpose(vol, (3, 0, 1, 2))
                vol = vol[0]
                inputs.append(vol)
                #labels = np.ones((1*self.batch_test,len(label_chunk[idx_filename])), dtype=int) * label_chunk[idx_filename]
                #targets.extend(labels)

            #targets = np.asarray(targets).reshape(1*self.batch_test,len(label_chunk[idx_filename]))
                
            inputs = np.asarray(inputs)
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs = np.asarray([data for data in data_ds]).reshape(first_dim, 128, 128, 22)
            yield inputs
            j = j + 1

def test_labels(self, test_ix):
    j = 0
    test_label_pool = [self.label_names[index] for index in test_ix]
    
    #transform to onehot
    onehot_encoder = OneHotEncoder(sparse_output=False)
    test_label_pool = np.array(test_label_pool).reshape(len(test_label_pool), 1)
    test_label_pool = onehot_encoder.fit_transform(test_label_pool)

    while True:
        if j * self.batch_test >= len(test_ix):  # This loop is used to run the generator indefinitely.
            j = 0
            # random.shuffle(self.subjects)
        else:
            label_chunk = test_label_pool[j * self.batch_test:(j + 1) * self.batch_test]
            targets = []
            for idx_filename in range(len(label_chunk)):
                labels = np.ones((620,len(label_chunk[idx_filename])), dtype=int) * label_chunk[idx_filename]
                targets.extend(labels)

            targets = np.asarray(targets).reshape(620*self.batch_test,len(label_chunk[idx_filename]))
           
            return targets
            j = j + 1   
            
def test_labels_short(self, test_ix):
    j = 0
    test_label_pool = [self.label_names[index] for index in test_ix]
    
    #transform to onehot
    onehot_encoder = OneHotEncoder(sparse_output=False)
    test_label_pool = np.array(test_label_pool).reshape(len(test_label_pool), 1)
    test_label_pool = onehot_encoder.fit_transform(test_label_pool)
    
    while True:
        if j * self.batch_test >= len(test_ix):  # This loop is used to run the generator indefinitely.
            j = 0
            # random.shuffle(self.subjects)
        else:
            label_chunk = test_label_pool[j * self.batch_test:(j + 1) * self.batch_test]
            targets = []
            for idx_filename in range(len(label_chunk)):
                labels = np.ones((1*self.batch_test,len(label_chunk[idx_filename])), dtype=int) * label_chunk[idx_filename]
                targets.extend(labels)

            targets = np.asarray(targets).reshape(1*self.batch_test*len(label_chunk),len(label_chunk[idx_filename]))
           
            yield targets
            j = j + 1   
            
def normalize(volume):
    numerator = tf.subtract(x=volume, y=tf.reduce_min(volume))
    denominator = tf.subtract(tf.reduce_max(volume), tf.reduce_min(volume))
    volume = tf.math.divide(numerator, denominator)
    return volume


def get_label_from_tsv_file(label_table, subj, sess):
    y = label_table.loc[label_table['participant_id'] == subj]
    if 'SIH' in set(y['condition']) and sess != 'ses-01':
        label = 1
    elif 'CFA' in set(y['condition']) and sess != 'ses-01':
        label = 2
    elif 'CPH' in set(y['condition']) and sess != 'ses-01':
        label = 3
    elif 'NAIVE' in set(y['condition']):
        label = 4
    else:
        label = 4  # SIH, CPH and CFA in ses-01 are naive
    return label
