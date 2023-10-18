import numpy as np

class Layer:
  def __init__ (self):
    pass

  def propagate (self):
    pass

  def backpropagate (self, diff, lr):
    pass

class DotLayer (Layer):
  def __init__ (self, x_generator, w_generator):
    super().__init__()
    self.x_generator = x_generator
    self.w_generator = w_generator

  def propagate (self):
    self.x_generator.propagate()
    self.w_generator.propagate()
    self.forward_res = self.x_generator.forward_res @ self.w_generator.forward_res

  def backpropagate (self, diff, lr):
    self.w_generator.backpropagate(np.expand_dims(self.x_generator.forward_res, axis=1) @ np.expand_dims(diff, axis=0), lr)
    self.x_generator.backpropagate(diff @ np.transpose(self.w_generator.forward_res), lr)

class AddLayer (Layer):
  def __init__ (self, x_generator, y_generator):
    super().__init__()
    self.x_generator = x_generator
    self.y_generator = y_generator
  
  def propagate (self):
    self.x_generator.propagate()
    self.y_generator.propagate()
    self.forward_res = self.x_generator.forward_res + self.y_generator.forward_res
  
  def backpropagate (self, diff, lr):
    self.x_generator.backpropagate(diff, lr)
    self.y_generator.backpropagate(diff, lr)

def generate_affine_generator (x_generator, w_generator, b_generator):
  dot_layer = DotLayer(x_generator, w_generator)
  add_layer = AddLayer(dot_layer, b_generator)
  return add_layer

class SigmoidLayer (Layer):
  def __init__ (self, x_generator):
    super().__init__()
    self.x_generator = x_generator

  def propagate (self):
    self.x_generator.propagate()
    self.forward_res = 1 / (1 + np.exp(-self.x_generator.forward_res))
  
  def backpropagate(self, diff, lr):
    self.x_generator.backpropagate(diff * self.forward_res * (1 - self.forward_res), lr)

class WeightLayer (Layer):
  def __init__ (self, w):
    super().__init__()
    self.w = w
  
  def propagate (self):
    self.forward_res = self.w
  
  def backpropagate(self, diff, lr):
    self.w -= lr * diff

class BiasLayer (Layer):
  def __init__ (self, b):
    super().__init__()
    self.b = b
  
  def propagate (self):
    self.forward_res = self.b
  
  def backpropagate(self, diff, lr):
    self.b -= lr * diff

class InputLayer (Layer):
  def __init__ (self, x):
    super().__init__()
    self.x = x

  def propagate (self):
    self.forward_res = self.x
  
  def backpropagate(self, diff, lr):
    pass

class MSELayer (Layer):
  def __init__ (self, x_generator, label):
    super().__init__()
    self.x_generator = x_generator
    self.label = label

  def propagate (self):
    self.x_generator.propagate()
    self.forward_res = 1 / 2 * np.sum(((self.x_generator.forward_res - self.label) ** 2))

  def backpropagate (self, diff, lr):
    self.x_generator.backpropagate((self.x_generator.forward_res - self.label) * diff, lr)
