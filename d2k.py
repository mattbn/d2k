
import xml.etree.ElementTree as xmlet
import numpy as np 
import tensorflow as tf 
from functools import reduce 

class D2k:
  
  def __init__(self):
    self.data = []
  #
  
  def from_string(self, s):
    validate = lambda x: x if x else ''
    self.data = list(
      map(
        lambda x:
          (x.attrib['type'], x.attrib, validate(x.text)) if not len(x) else 
          (x[0].tag, {**x.attrib, **x[0].attrib}, validate(x[0].text)), 
        xmlet.fromstring(s)[::-1]
      )
    )
    return self
  #
  
  def from_file(self, path):
    with open(path) as f:
      return self.from_string(''.join(f.readlines()))
  #
  
  def convert(self, **model_kwargs):
    losses = {'loss_multiclass_log' : 'SparseCategoricalCrossentropy'}
    # data will be a list of [layers, losses]
    self.data = [ # the first element is a list of (KerasTensor, layer name)
      reduce( # L: mapped layers, x: current layer(unmapped)
        lambda L,x: L + getattr(Layers, x[0]+'_')(*x[1:])(L), 
        filter( # y: current layer (unmapped)
          lambda y: y[0]+'_' in Layers.__dict__, 
          self.data
        ), 
        []
      ), 
      list( # the second element is a list of losses
        map(
          lambda x: losses[x[0]], 
          filter(
            lambda y: y[0] in losses, 
            self.data
          )
        )
      )
    ]
    
    # build the model
    self.data[0] = tf.keras.Model(
      list(
        map(
          lambda x: x[0], 
          filter(lambda y: y[1] == 'Input', self.data[0])
        )
      ), 
      [self.data[0][-1][0]], 
      **model_kwargs
    )
    
    return self
  #
#

class Layers:
  
  def get_ref_id(layers, rid):
    tags = list(
      map(
        lambda x: len(layers)-1-x[0], 
        filter(
          lambda y: y[1][1] == 'Tag' and y[1][0].__dict__['tag'] == rid, 
          enumerate(layers[::-1])
        )
      )
    )
    return tags[0] if len(tags) else -1
  #
  
  ##############
  ##  layers  ##
  ##############
  
  def input_rgb_image_(*args, **kwargs):
    def input_rgb_image__(*args, **kwargs):
      return [(
        tf.keras.Input(
          shape=(None, None, 3)
        ), 
        'Input'
      )]
    #
    return input_rgb_image__
  #
  
  def input_rgb_image_sized_(attrib, *args, **kwargs):
    def input_rgb_image_sized__(*args, **kwargs):
      return [(
        tf.keras.Input(
          shape=(
            int(attrib['nr']), 
            int(attrib['nc']), 
            3
          )
        ), 
        'Input'
      )]
    #
    return input_rgb_image_sized__
  #
  
  def relu_(*args, **kwargs):
    def relu__(layers, *args, **kwargs):
      return [(tf.keras.layers.ReLU()(layers[-1][0]), 'ReLU')]
    #
    return relu__
  #
  
  def softmax_(*args, **kwargs):
    def softmax__(layers, *args, **kwargs):
      return [(tf.keras.layers.Softmax()(layers[-1][0]), 'Softmax')]
    #
    return softmax__
  #
  
  def multiply_(attrib, *args, **kwargs):
    def multiply(layers, *args, **kwargs):
      return [(
        tf.keras.layers.Rescaling(
          scale=float(attrib['val'])
        )(layers[-1][0]), 
        'Rescaling'
      )]
    #
    return multiply__
  #
  
  def tag_(attrib, *args, **kwargs):
    def tag__(layers, *args, **kwargs):
      res = tf.keras.layers.Layer()(layers[-1][0])
      res.__dict__['tag'] = int(attrib['id'])
      return [(res, 'Tag')]
    #
    return tag__
  #
  
  def skip_(attrib, *args, **kwargs):
    def skip__(layers, *args, **kwargs):
      return [(
        tf.keras.layers.Layer()(
          layers[Layers.get_ref_id(layers, int(attrib['id']))]
        ), 
        'Skip'
      )]
    #
    return skip__
  #
  
  def add_prev_(attrib, *args, **kwargs):
    def add_prev__(layers, *args, **kwargs):
      return [()]
    return add_prev__
  #
  
  def affine_con_(attrib, str_weights, *args, **kwargs):
    def affine_con__(layers, *args, **kwargs):
      layer = tf.keras.layers.LayerNormalization()
      res = layer(layers[-1][0])
      layer.set_weights(
        np.array([
          np.array(str_weights[:-len(str_weights)//2]).astype(np.float32), 
          np.array(str_weights[len(str_weights)//2:]).astype(np.float32)
        ], dtype=object)
      )
      return [(res, 'LayerNormalization')]
    #
    return affine_con__
  #
  
  def avg_pool_(attrib, *args, **kwargs):
    def avg_pool__(layers, *args, **kwargs):
      res = [(
        tf.keras.layers.ZeroPadding2D(
          padding=(int(attrib['padding_x']), int(attrib['padding_y']))
        )(layers[-1][0]), 
        'ZeroPadding2D'
      )]
      res += [(
        tf.keras.layers.AveragePooling2D(
          padding='valid', 
          pool_size=(int(attrib['nr']), int(attrib['nc'])), 
          strides=(int(attrib['stride_x']), int(attrib['stride_y']))
        )(res[-1][0]), 
        'AveragePooling2D'
      )]
      return res
    #
    def global_avg_pool__(layers, *args, **kwargs):
      return [(
        tf.keras.layers.GlobalAveragePooling2D(
          keepdims=True
        )(layers[-1][0]), 
        'GlobalAveragePooling2D'
      )]
    #
    if int(attrib['nr']) and int(attrib['nc']):
      return avg_pool__
    return global_avg_pool__
  #
  
  def max_pool_(attrib, *args, **kwargs):
    def max_pool__(layers, *args, **kwargs):
      res = [(
        tf.keras.layers.ZeroPadding2D(
          padding=(int(attrib['padding_x']), int(attrib['padding_y']))
        )(layers[-1][0]), 
        'ZeroPadding2D'
      )]
      res += [(
        tf.keras.layers.MaxPooling2D(
          padding='valid', 
          pool_size=(int(attrib['nr']), int(attrib['nc'])), 
          strides=(int(attrib['stride_x']), int(attrib['stride_y']))
        )(res[-1][0]), 
        'MaxPooling2D'
      )]
      return res
    #
    def global_max_pool__(layers, *args, **kwargs):
      return [(
        tf.keras.layers.GlobalMaxPooling2D(
          keepdims=True
        )(layers[-1][0]), 
        'GlobalMaxPooling2D'
      )]
    #
    if int(attrib['nr']) and int(attrib['nc']):
      return max_pool__
    return global_max_pool__
  #
  
  def fc_(attrib, str_weights, *args, **kwargs):
    str_weights = list(
      filter(
        lambda x: x, 
        str_weights.split('\n')
      )
    )
    
    def fc__(layers, *args, **kwargs):
      res = [(tf.keras.layers.Flatten()(layers[-1][0]), 'Flatten')]
      layer = tf.keras.layers.Dense(
        units=int(attrib['num_outputs']), 
        use_bias=True if attrib['use_bias'] == 'true' else False
      )
      res += [(layer(res[-1][0]), 'Dense')]
      layer.set_weights(
        np.array(
          [
            np.array(
              list(map(lambda x: x.split(), str_weights))
            )[:-1].reshape(
              np.array(res[-2][0].shape)[1], 
              int(attrib['num_outputs'])
            ).astype(np.float32). 
            np.array(str_weights[-1].split()).astype(np.float32)
          ], 
          dtype=object
        ) if 'use_bias' in attrib and attrib['use_bias'] == 'true' else
        np.array(
          np.array(list(map(lambda x: x.split(), str_weights))), 
          dtype=object
        ).astype(np.float32)
      )
      return res
    #
    return fc__
  #
  
  def con_(attrib, str_weights, *args, **kwargs):
    str_weights = str_weights.split()
    def con__(layers, *args, **kwargs):
      res = [(
        tf.keras.layers.ZeroPadding2D(
          padding=(int(attrib['padding_x']), int(attrib['padding_y']))
        )(layers[-1][0]), 
        'ZeroPadding2D'
      )]
      layer = tf.keras.layers.Conv2D(
        padding='valid', 
        filters=int(attrib['num_filters']), 
        kernel_size=(int(attrib['nr']), int(attrib['nc'])), 
        strides=(int(attrib['stride_x']), int(attrib['stride_y']))
      )
      res += [(layer(res[-1][0]), 'Conv2D')]
      layer.set_weights(
        np.array(
          [
            np.array(str_weights[:-int(attrib['num_filters'])]).reshape(
              int(attrib['nr']), 
              int(attrib['nc']), 
              np.array(layers[-1][0].shape)[3], 
              int(attrib['num_filters'])
            ).astype(np.float32), 
            np.array(str_weights[-int(attrib['num_filters']):]).astype(
              np.float32
            )
          ], 
          dtype=object
        ) if 'use_bias' in attrib and attrib['use_bias'] == 'true' else 
        np.array(str_weights, dtype=object).reshape(
          int(attrib['nr']), 
          int(attrib['nc']), 
          np.array(layers[-1][0].shape)[3], 
          int(attrib['num_filters'])
        ).astype(np.float32)
      )
      return res
    #
    return con__
  #
#
