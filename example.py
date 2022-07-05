
from d2k import *
from tensorflow.keras.utils import plot_model

m = D2k().from_file(
  input('path: ')
).convert().data

m.summary()
input('')
print('plotting to output.png...')
plot_model(m, to_file='output.png', show_shapes=True)
