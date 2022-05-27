
import json
import xml.etree.ElementTree as xmlet

from copy import deepcopy
from functools import reduce

import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *
#import tensorflow_addons as tfa


class D2k:
  
  def load_mappings(path):
    with open(path) as f:
      return json.loads(''.join(f.readlines()))
  #
  
  def __init__(self, mappings=load_mappings('mappings.json')):
    self.mappings = mappings
    self.data = None
  #
  
  def from_string(self, text):
    validate = lambda x: x if x else ''
    self.data = list(
      map(
        lambda x: 
          (x.attrib['type'], x.attrib, validate(x.text)) if not len(x) else 
          (x[0].tag, {**x.attrib, **x[0].attrib}, validate(x[0].text)), 
        xmlet.fromstring(text)[::-1]
      )
    )
    return self
  #
  
  def from_file(self, path):
    with open(path) as f:
      return self.from_string(''.join(f.readlines()))
  #
  
  def convert(self):
    def map_attribute(name, attrib, attr, idx, kname, kattrib, data, maps):
      dmaps = maps['d']['attrib'][name]
      kmaps = maps['k']['attrib'][kname]
      cast = lambda x:x if x != str(x) else int(x) if x.isdigit() else float(x)
      
      # check if there is a mapping for that attribute
      if attr in dmaps and dmaps[attr] in kmaps:
        if dmaps[attr] in kattrib: # attribute was already mapped
          # then it's a multi-dimensional attribute => should be a tuple
          if type(kattrib[dmaps[attr]]) != type(()):
            kattrib[dmaps[attr]] = (kattrib[dmaps[attr]],)
          if attr in {'nr','nc'}: # special case
            kattrib[dmaps[attr]] = (*kattrib[dmaps[attr]], cast(attrib[attr]))
          else:
            kattrib[dmaps[attr]] = (cast(attrib[attr]), *kattrib[dmaps[attr]])
        
        else: # attribute is being mapped for the first time
          kattrib[dmaps[attr]] = cast(attrib[attr])
      
      return kattrib
    #
    
    def load_weights(str_weights, delta):
      str_weights = str_weights.strip().split('\n')
      weights = [[], []] # [ weights, biases ]
      
      # load columns as weight arrays
      for i in range(0, len(str_weights[0].split())):
        for j,ln in enumerate(str_weights):
          k = 0 if j < len(str_weights)-delta else 1
          if i == len(weights[k]):
            weights[k] += [[]]
          weights[k][i] += [float(ln.split()[i])]
      return weights
    #
    
    D,K = self.mappings['d'], self.mappings['k']
    
    # extract losses
    losses = [D['losses'][l[0]] for l in self.data if l[0] in D['losses']]
    layers = list(filter(lambda x: x[0] in D['layers'], self.data))
    
    self.data = [[], {}, losses]
    
    for l in layers:
      names, attrib = deepcopy(D['layers'][l[0]]), {}
      
      delta = 0
      if 'use_bias' in l[1]:
        if 'fc' == l[0]:
          delta = 1
        elif 'con' == l[0]:
          delta = int(l[1]['num_filters'])
      weights = load_weights(l[2], delta)
      
      for i,n in enumerate(names):
        # calls map_attribute for each attribute in l[1]
        attrib = reduce(
          lambda A, k: # A: mapped attributes, k: attribute names iterator
            map_attribute(l[0], l[1], k, i, n, A, self.data, self.mappings), 
          l[1], 
          {}
        )
        
        # special cases
        if n == 'Input':
          if 'shape' not in attrib:
            attrib['shape'] = (None, None, 3)
          elif len(attrib['shape']) < 3:
            attrib['shape'] = (*attrib['shape'], 3)
        elif n in {'Conv2D', 'AveragePooling2D', 'MaxPooling2D'}:
          attrib['padding'] = 'valid'
        if 'pool_size' in attrib and type(attrib['pool_size']) == type(()):
          attrib['pool_size'] = tuple(
            x if x else 1 for x in attrib['pool_size']
          )
        
        # generate edges
        self.data[1][len(self.data[0])] = [] # edge list
        if n != 'Input':
          if 'tag' in attrib:
            if l[0] != 'tag': # skip or layer that references a tag
              for j,m in enumerate(self.data[0][::-1]):
                if 'tag' in m[1] and m[1]['tag'] == attrib['tag']:
                  self.data[1][len(self.data[0])] += [len(self.data[0])-1-j]
                  break
          if l[0] != 'skip': # tag or layer that references a tag
            self.data[1][len(self.data[0])] += [len(self.data[0])-1]
        
        # remove biases if not used
        if len(weights) and not len(weights[1]):
          weights = weights[0]
        # only last layer gets the weights (?)
        self.data[0] += [(n, attrib, weights if i == len(names)-1 else [])]
    
    # remove tag attributes
    for tp in self.data[0]:
      if 'tag' in tp[1]:
        tp[1].pop('tag')
    
    return self
  #
  
  def build(self, **model_kwargs):
    
    # instantiate classes from class names
    knodes = list(map(lambda n: globals()[n[0]](**n[1]), self.data[0]))
    inputs, outputs, y = [], [], deepcopy(knodes)
    nodes, edges = self.data[0], self.data[1]
    
    for idx in edges:
      if nodes[idx][0] != 'Input':
        # get output of all layers that connect to the current one
        layer_inputs = [y[e] for e in edges[idx]]
        # get indices for all layers this one connects to
        layer_outputs = [x for e in edges for x in edges[e] if x == idx]
        
        if len(layer_inputs) > 1: # layer call expects a list
          y[idx] = knodes[idx](layer_inputs)
        else: # layer call does not expect a list
          y[idx] = knodes[idx](*layer_inputs)
        
        # set weights
        knodes[idx].set_weights(np.array(self.data[0][idx][2]))
        
        if not len(layer_outputs): # no layer uses this one's outputs
          outputs += [y[idx]] # it's an output
      else: # it's an input
        inputs += [knodes[idx]]
    
    # build model
    model = Model(inputs, outputs, **model_kwargs)
    self.data = [inputs, outputs, model, self.data[2]] # keep losses
    return self
  #
#
