import tensorflow as tf
import numpy as np

from model import poly2poly

sess = tf.Session()
model = poly2poly(sess, 'archpoly2poly', '../../../data/processed/train/',
                  '../../../data/processed/test/',
                  '../../../data/processed/val/')

model.train(10)
