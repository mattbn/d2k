
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
    # get tags with id=rid in reverse order
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
    def multiply__(layers, *args, **kwargs):
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
      res = tf.keras.layers.Layer(
        name='tag_' + attrib['id'] + '_' + attrib['idx']
      )(layers[-1][0])
      res.__dict__['tag'] = int(attrib['id'])
      return [(res, 'Tag')]
    #
    return tag__
  #
  
  # outputs the 'id' layer's output
  def skip_(attrib, *args, **kwargs):
    def skip__(layers, *args, **kwargs):
      return [(
        tf.keras.layers.Layer(
          name='skip_' + attrib['id'] + '_' + attrib['idx']
        )(
          layers[Layers.get_ref_id(layers, int(attrib['id']))][0]
        ), 
        'Skip'
      )]
    #
    return skip__
  #
  
  def add_prev_(attrib, *args, **kwargs):
    def add_prev__(layers, *args, **kwargs):
      res, prev = [], layers[-1][0]
      ref = layers[Layers.get_ref_id(layers, int(attrib['tag']))][0]
      
      def size_shape(delta, tensors, layer, name, *params):
        list_or_tuple = lambda *p: [*p] if len(p) > 1 else p[0]
        r = [(
          layer(list_or_tuple(
            tensors[0] if delta < 0 else tensors[1], 
            *params
          )), 
          name
        )] if delta else []
        if delta < 0:
          tensors = (r[0][0], tensors[1])
        elif delta > 0:
          tensors = (tensors[0], r[0][0])
        # returns:
        # 1 - updated layer list
        # 2 - updated prev, ref tensors
        # 3 - tuple of shape differences between prev and ref
        return res + r, *tensors, tuple(x-y if x and y else 0 for x,y in zip(
          *(
            # tensor shapes
            tuple(map(lambda x: x if x else 0, tensors[0].shape)), 
            tuple(map(lambda x: x if x else 0, tensors[1].shape))
          )
        ))
      
      delta = 0
      res, prev, ref, delta = size_shape(delta, (prev, ref), None, '')
      
      if delta[1]: # delta_x != 0
        res, prev, ref, delta = size_shape(
          delta[1], (pref, ref), 
          tf.keras.layers.ZeroPadding2D(
            padding=((0, abs(delta[0])), (0, 0))
          ), 
          'ZeroPadding2D', 
        )
      
      if delta[2]: # delta_y != 0
        res, prev, ref, delta = size_shape(
          delta[2], (prev, ref), 
          tf.keras.layers.ZeroPadding2D(
            padding=((0, 0), (0, abs(delta[1])))
          ), 
          'ZeroPadding2D'
        )
      
      if delta[3]: # delta_z != 0
        # shape difference
        shape_delta = prev.shape[1:-1] if delta[3] < 0 else ref.shape[1:-1]
        shape_delta += [abs(delta[3])]
        
        res, prev, ref, delta = size_shape(
          delta[3], (prev, ref), 
          tf.keras.layers.Concatenate(axis=3), 
          'Concatenate', 
          tf.keras.layers.Lambda(lambda x: x)(
            tf.zeros_like(
              tf.Variable(
                [[[[0 for x in range(0, shape_delta[2])]]]], 
                shape=[None, None, None, shape_delta[2]]
              ), 
              dtype=tf.float32
            )
          )
          #tf.keras.Input(tensor=tf.zeros_like(
          #  tf.placeholder(
          #    tf.float32, 
          #    shape_delta
          #  )
          #))
        )
      
      return res + [(
        tf.keras.layers.Add()([prev, ref]), 
        'Add'
      )]
    #
    return add_prev__
  #
  
  def affine_con_(attrib, str_weights, *args, **kwargs):
    str_weights = str_weights.split()
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
      res, prev = [], layers[-1][0]
      # add padding
      if int(attrib['padding_x']) or int(attrib['padding_y']):
        res += [(
          tf.keras.layers.ZeroPadding2D(
            padding=(int(attrib['padding_x']), int(attrib['padding_y']))
          )(prev), 
          'ZeroPadding2D'
        )]
        prev = res[-1][0]
      
      res += [(
        tf.keras.layers.AveragePooling2D(
          padding='valid', 
          pool_size=(int(attrib['nr']), int(attrib['nc'])), 
          strides=(int(attrib['stride_x']), int(attrib['stride_y']))
        )(prev), 
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
      res, prev = [], layers[-1][0]
      # add padding
      if int(attrib['padding_x']) or int(attrib['padding_y']):
        res += [(
          tf.keras.layers.ZeroPadding2D(
            padding=(int(attrib['padding_x']), int(attrib['padding_y']))
          )(prev), 
          'ZeroPadding2D'
        )]
        prev = res[-1][0]
      
      res += [(
        tf.keras.layers.MaxPooling2D(
          padding='valid', 
          pool_size=(int(attrib['nr']), int(attrib['nc'])), 
          strides=(int(attrib['stride_x']), int(attrib['stride_y']))
        )(prev), 
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
            ).astype(np.float32), 
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
      res, prev = [], layers[-1][0]
      # add padding
      if int(attrib['padding_x']) or int(attrib['padding_y']):
        res += [(
          tf.keras.layers.ZeroPadding2D(
            padding=(int(attrib['padding_x']), int(attrib['padding_y']))
          )(layers[-1][0]), 
          'ZeroPadding2D'
        )]
        prev = res[-1][0]
      
      layer = tf.keras.layers.Conv2D(
        padding='valid', 
        filters=int(attrib['num_filters']), 
        kernel_size=(int(attrib['nr']), int(attrib['nc'])), 
        strides=(int(attrib['stride_x']), int(attrib['stride_y']))
      )
      res += [(layer(prev), 'Conv2D')]
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
