
import xml.etree.ElementTree as xmlet
import numpy as np 
import tensorflow as tf 
from functools import reduce 
import enum

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
    losses = {'loss_multiclass_log' : 'sparse_categorical_crossentropy'}
    # data will be a list of [tensors, losses]
    self.data = [ # the first element is a list of KerasTensor s
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
      list( # get all tensors with layer_type == Input
        map(
          lambda x: Layers.remove_tensor_type(x), 
          filter(
            lambda y: 'layer_type' in y.__dict__ and (
              y.__dict__['layer_type'] == Layers.Type.Input
            ), 
            self.data[0]
          )
        )
      ), 
      list( # get all tensors with layer_type == Output
        map(
          lambda x: Layers.remove_tensor_type(x), 
          filter(
            lambda y: 'layer_type' in y.__dict__ and (
              y.__dict__['layer_type'] == Layers.Type.Output
            ), 
            self.data[0]
          )
        )
      ), 
      **model_kwargs
    )
    
    return self
  #
#

class Layers:
  
  class Type(enum.Enum):
    Input = 0
    Hidden = 1
    Output = 2
  #
  
  # gets the last tensor with tag == rid
  def get_ref(tensors, rid):
    for t in tensors[::-1]:
      if 'tag' in t.__dict__ and t.__dict__['tag'] == rid:
        return t
    raise RuntimeError('get_ref: layer with tag ' + rid + 'was not found.')
  #
  
  # adds a layer type to tensor (used to track inputs and outputs)
  def add_tensor_type(tensor, value):
    tensor.__dict__['layer_type'] = value
    return tensor
  #
  
  # removes layer type from tensor
  # called when outputs need to be updated 
  # (when a layer is no longer an output)
  def remove_tensor_type(tensor):
    if 'layer_type' in tensor.__dict__:
      if tensor.__dict__['layer_type'] != Layers.Type.Input:
        del tensor.__dict__['layer_type']
    return tensor
  #
  
  ##############
  ##  layers  ##
  ##############
  
  # RGB image with unknown size
  def input_rgb_image_(attrib, *args, **kwargs):
    def input_rgb_image__(*args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.Input(shape=(None, None, 3)), 
        Layers.Type.Input
      )]
    #
    return input_rgb_image__
  #
  
  def input_rgb_image_sized_(attrib, *args, **kwargs):
    def input_rgb_image_sized__(*args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.Input(
          shape=(int(attrib['nc']), int(attrib['nr']), 3)
        ), 
        Layers.Type.Input
      )]
    #
    return input_rgb_image_sized__
  #
  
  def relu_(*args, **kwargs):
    def relu__(tensors, *args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.layers.ReLU()(Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
    #
    return relu__
  #
  
  def softmax_(*args, **kwargs):
    def softmax__(tensors, *args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.layers.Softmax()(Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
    #
    return softmax__
  #
  
  def multiply_(attrib, *args, **kwargs):
    def multiply__(tensors, *args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.layers.Rescaling(scale=float(attrib['val']))(
          Layers.remove_tensor_type(tensors[-1])
        ), 
        Layers.Type.Output
      )]
    #
    return multiply__
  #
  
  def tag_(attrib, *args, **kwargs):
    def tag__(tensors, *args, **kwargs):
      # nop; only adds a tag to the last tensor
      tensors[-1].__dict__['tag'] = int(attrib['id'])
      return []
    #
    return tag__
  #
  
  def skip_(attrib, *args, **kwargs):
    def skip__(tensors, *args, **kwargs):
      ref = Layers.get_ref(tensors, int(attrib['id']))
      del ref.__dict__['tag'] # remove tag
      Layers.remove_tensor_type(tensors[-1])
      return [Layers.add_tensor_type(
        tf.keras.layers.Layer()(ref), 
        Layers.Type.Output
      )] # output tagged layer's output
    #
    return skip__
  #
  
  def add_prev_(attrib, *args, **kwargs):
    def add_prev__(tensors, *args, **kwargs):
      inputs = [
        Layers.remove_tensor_type(tensors[-1]), # last tensor
        Layers.get_ref(tensors, int(attrib['tag'])) # tagged tensor
      ]
      
      del inputs[1].__dict__['tag'] # remove tag
      # tuple of shape differences
      delta = tuple(
        abs(x-y) if x and y else 0 for x,y in zip(
          inputs[0].shape, inputs[1].shape
        )
      )
      
      # there is at least 1 dim that should be padded
      if delta > tuple(0 for x in range(len(delta))):
        def zero_pad_tensors(ts):
          # gets the max shape in ts, then zero pads every tensor
          # with the difference between max shape and ts[i].shape
          get_valid = lambda t: tuple(x if x else 0 for x in t)
          max_shape = tuple(
            max(get_valid(tp)) for tp in zip(*tuple(t.shape for t in ts))
          )
          
          return [tf.pad(
            Layers.remove_tensor_type(t), 
            list(
              map(
                # (0, shape dim difference) if dim != None else (0, 0)
                # x == tuple(dim index, dim)
                lambda x: (0, x[1]-tf.shape(t)[x[0]]) if x[1] else (0, 0), 
                enumerate(max_shape)
              )
            )
          ) for t in ts]
        #
        inputs = tf.keras.layers.Lambda(zero_pad_tensors)(inputs)
      
      return [Layers.add_tensor_type(
        tf.keras.layers.Add()(inputs), 
        Layers.Type.Output
      )]
    #
    return add_prev__
  #
  
  def affine_con_(attrib, str_weights, *args, **kwargs):
    str_weights = str_weights.split() # get list of lines (one weight/line)
    def affine_con__(tensors, *args, **kwargs):
      # BatchNormalization should behave like affine_con when training=False 
      # and (moving_mean, moving_variance) = (0, 1)
      # Its formula is:
      #                input - moving_mean
      # gamma * --------------------------------- + beta
      #          sqrt(moving_variance + epsilon)
      # 
      layer = tf.keras.layers.BatchNormalization(epsilon=0)
      result = layer(Layers.remove_tensor_type(tensors[-1]))
      if len(str_weights): # there are weights
        layer.set_weights(
          # input weights order is (gamma, beta)
          np.array([ # weights order is: (gamma, beta, mean, var)
            np.array(str_weights[:-len(str_weights)//2]).astype(np.float32), 
            np.array(str_weights[len(str_weights)//2:]).astype(np.float32), 
            np.full((tensors[-1].shape[-1]), 0), 
            np.full((tensors[-1].shape[-1]), 1)
          ], dtype=object)
        )
      return [Layers.add_tensor_type(result, Layers.Type.Output)]
    #
    return affine_con__
  #
  
  def avg_pool_(attrib, *args, **kwargs):
    def avg_pool__(tensors, *args, **kwargs):
      padding = 'valid'
      
      # add padding ?
      if int(attrib['padding_x']) or int(attrib['padding_y']):
        padding = 'same'
      
      return [Layers.add_tensor_type(
        tf.keras.layers.AveragePooling2D(
          padding=padding, 
          pool_size=(int(attrib['nr']), int(attrib['nc'])), 
          strides=(int(attrib['stride_y']), int(attrib['stride_x']))
        )(Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
    #
    def global_avg_pool__(tensors, *args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.layers.GlobalAveragePooling2D(
          keepdims=True
        )(Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
    #
    
    # pool size dim == 0 means that dim == input dim
    if int(attrib['nr']) and int(attrib['nc']):
      return avg_pool__
    return global_avg_pool__
  #
  
  def max_pool_(attrib, *args, **kwargs):
    def max_pool__(tensors, *args, **kwargs):
      padding = 'valid'
      
      # add padding ? 
      if int(attrib['padding_x']) or int(attrib['padding_y']):
        padding = 'same'
      
      return res + [Layers.add_tensor_type(
        tf.keras.layers.MaxPooling2D(
          padding=padding, 
          pool_size=(int(attrib['nr']), int(attrib['nc'])), 
          strides=(int(attrib['stride_y']), int(attrib['stride_x']))
        )(res[-1] if res else Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
    #
    def global_max_pool__(tensors, *args, **kwargs):
      return [Layers.add_tensor_type(
        tf.keras.layers.GlobalMaxPooling2D(
          keepdims=True
        )(Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
    #
    
    # pool size dim == 0 means that dim == input dim
    if int(attrib['nr']) and int(attrib['nc']):
      return max_pool__
    return global_max_pool__
  #
  
  def fc_(attrib, str_weights, *args, **kwargs):
    # split lines and remove empty lists
    str_weights = list(filter(lambda x: x, str_weights.split('\n')))
    def fc__(tensors, *args, **kwargs):
      # Dense inputs should be always Flatten outputs
      res = [
        tf.keras.layers.Flatten()(Layers.remove_tensor_type(tensors[-1]))
      ]
      
      use_bias_flag = True
      if 'use_bias' in attrib and attrib['use_bias'] != 'true':
        use_bias_flag = False
      layer = tf.keras.layers.Dense(
        units=int(attrib['num_outputs']), 
        use_bias=use_bias_flag
      )
      res += [Layers.add_tensor_type(layer(res[-1]), Layers.Type.Output)]
      if len(str_weights): # there are weights
        if use_bias_flag: # there are biases in weights
          try:
            layer.set_weights(
              np.array(
                [
                  np.array(
                    list(map(lambda x: x.split(), str_weights[:-1]))
                  ).reshape(
                    np.array(res[-2].shape)[1], 
                    int(attrib['num_outputs'])
                  ).astype(np.float32), 
                  np.array(str_weights[-1].split()).astype(np.float32)
                ], dtype=object
              )
            )
          except BaseException: # does not have biases or an exc was raised
            # try creating layer without biases
            use_bias_flag = False
            layer.use_bias = False
            res[-1] = Layers.add_tensor_type(
              layer(res[-2]), 
              Layers.Type.Output
            )
        
        if not use_bias_flag: # no biases in weights
          layer.set_weights(
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
    str_weights = str_weights.split() # remove '\n'
    def con__(tensors, *args, **kwargs):
      padding = 'valid'
      use_bias_flag = True
      if 'use_bias' in attrib and attrib['use_bias'] != 'true':
        use_bias_flag = False
      
      # add padding ? 
      if int(attrib['padding_x']) or int(attrib['padding_y']):
        padding = 'same'
      
      layer = tf.keras.layers.Conv2D(
        padding=padding, 
        filters=int(attrib['num_filters']), 
        kernel_size=(int(attrib['nr']), int(attrib['nc'])), 
        strides=(int(attrib['stride_y']), int(attrib['stride_x'])), 
        use_bias=use_bias_flag
      )
      # insert Conv2D in output list
      res = [Layers.add_tensor_type(
        layer(Layers.remove_tensor_type(tensors[-1])), 
        Layers.Type.Output
      )]
      
      if len(str_weights): # there are weights
        if use_bias_flag: # there are biases in weights
          try:
            layer.set_weights(
              np.array(
                [
                  np.transpose(
                    np.array(
                      str_weights[:-int(attrib['num_filters'])]).reshape(
                        int(attrib['nr']), 
                        int(attrib['nc']), 
                        np.array(tensors[-1].shape)[3], 
                        int(attrib['num_filters'])
                      ).astype(np.float32), 
                      [2, 3, 1, 0]
                  ), 
                  np.array(str_weights[-int(attrib['num_filters']):]).astype(
                    np.float32
                  )
                ], dtype=object
              )
            )
          except BaseException: # does not have biases or an exc was raised
            # try creating layer without biases
            use_bias_flag = False
            layer = tf.keras.layers.Conv2D(
              padding=padding, 
              filters=int(attrib['num_filters']), 
              kernel_size=(int(attrib['nr']), int(attrib['nc'])), 
              strides=(int(attrib['stride_y']), int(attrib['stride_x'])), 
              use_bias=False
            )
            res[0] = Layers.add_tensor_type(
              layer(Layers.remove_tensor_type(tensors[-1])), 
              Layers.Type.Output
            )
        
        if not use_bias_flag: # no biases in weights
          layer.set_weights(
            np.transpose(
              np.array(str_weights, dtype=object).reshape(
                int(attrib['nr']), 
                int(attrib['nc']), 
                np.array(tensors[-1].shape)[3], 
                int(attrib['num_filters'])
              ).astype(np.float32), 
              [2, 3, 1, 0]
            )
          )
      return res
    #
    return con__
  #
#
