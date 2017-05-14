import tensorflow as tf
import numpy as np

from model import poly2poly

sess = tf.Session()
model = recon2recon(sess, 'archpoly2poly', '../../../data/processed/train/',
                    '../../../data/processed/test/',
                    '../../../data/processed/val/')

model.train(10)
