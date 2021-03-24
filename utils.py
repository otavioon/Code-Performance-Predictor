import yaml
import numpy as np
import os

from tensorflow import keras
from spektral.layers import *

# Some helper functions
def yaml_load(filename: str) -> dict:
    with open(filename, 'rt') as f:
        return yaml.load(f, Loader=yaml.CLoader)
    
def yaml_write(filename: str, obj, mode='wt'):
    with open(filename, mode) as f:
        yaml.dump(obj, f)

# Very bad way, but faster
def get_section(filename: str, section: str):
    lines = []
    with open(filename) as f:
        count = False

        for line in f.readlines():
            if line.startswith(section):
                count = True
                continue

            if count:
                if str.isspace(line[0]):
                    lines.append(line)
                else:
                    return lines
        return lines
    
class GraphDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_root: str, filenames: list, labels: dict, batch_size: int=16, shuffle: bool=True):
        'Initialization'
        self.data_root = data_root
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of files
        files = [self.filenames[idx] for idx in indexes]

        # Generate data
        graphs, features, speedups = self.__data_generation(files)

        return [features[:,0,:,:], graphs[:,0,:,:], features[:,1,:,:], graphs[:,1,:,:]], [speedups[:,0:4], speedups[:,4]]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        graphs = []
        features = []
        labels = []
        
        for f in list_files:
            filename = os.path.join(self.data_root, f'{f}.npz')
            np_loaded = np.load(filename, allow_pickle=True)
            graphs.append(np_loaded['graph'])
            features.append(np_loaded['feature'])
            
            label = self.labels[f]
            y_c = np.zeros(5)
            if label <= 0.996:
                y_c[0]=1
            elif label > 0.996 and label < 1.0:
                y_c[1]=1
            elif (label >= 1.0 and label < 1.15):
                y_c[2]=1
            elif (label > 1.15):
                y_c[3]= 1
            y_c[4] = label
            labels.append(y_c)

        graphs = np.stack(graphs)
        features = np.stack(features)
        labels = np.stack(labels)
        
        graphs[:,0,:,:] = GraphConv.preprocess(graphs[:,0,:,:]).astype('f4')
        graphs[:,1,:,:] = GraphConv.preprocess(graphs[:,1,:,:]).astype('f4')
        
        return graphs, features, labels
