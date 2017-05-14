import pickle
import h5py

from utils import make_data_multi
import sys
import os

train_path = '../../data/processed/train_multi/'
val_path = '../../data/processed/val_multi/'
test_path = '../../data/processed/test_multi/'

os.mkdir(train_path)
os.mkdir(val_path)
os.mkdir(test_path)


# make train
for i in range(500):
    recon, img = make_data_multi()
    f = h5py.File(train_path + "{}.hdf5".format(i), "w")
    f.create_dataset('recon', data=recon)
    f.create_dataset('img', data=img)
    f.close()

# make validation
for i in range(100):
    recon, img = make_data_multi()
    f = h5py.File(val_path + "{}.hdf5".format(i), "w")
    f.create_dataset('recon', data=recon)
    f.create_dataset('img', data=img)
    f.close()

# make test
for i in range(500):
    recon, img = make_data_multi()
    f = h5py.File(test_path + "{}.hdf5".format(i), "w")
    f.create_dataset('recon', data=recon)
    f.create_dataset('img', data=img)
    f.close()
