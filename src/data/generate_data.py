import pickle
import h5py

from utils import make_data
import sys
import os

train_path = '../data/processed/train'
val_path = '../data/processed/val'
test_path = '../data/processed/test'

os.mkdir(train_path)
os.mkdir(val_path)
os.mkdir(test_path)


# make train
for i in range(500):
    recon, min_diff, mid_energy = make_data()
    f = h5py.File(train_path + "{}.hdf5".format(i), "w")
    f.create_dataset('recon', data=lim)
    f.create_dataset('min_diff', data=full)
    f.create_dataset('mid_energy', data=full)
    f.close()

# make validation
for i in range(100):
    recon, min_diff, mid_energy = make_data()
    f = h5py.File(val_path + "{}.hdf5".format(i), "w")
    f.create_dataset('recon', data=lim)
    f.create_dataset('min_diff', data=full)
    f.create_dataset('mid_energy', data=full)
    f.close()

# make test
for i in range(500):
    recon, min_diff, mid_energy = make_data()
    f = h5py.File(test_path + "{}.hdf5".format(i), "w")
    f.create_dataset('recon', data=lim)
    f.create_dataset('min_diff', data=full)
    f.create_dataset('mid_energy', data=full)
    f.close()
