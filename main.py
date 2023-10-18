from layer import *

weights_sd = 0.01
weights = [
  np.random.randn(2, 32) * np.sqrt(2. / 2),
  np.random.randn(32, 32) * np.sqrt(2. / 32),
  np.random.randn(32, 32) * np.sqrt(2. / 32),
  np.random.randn(32, 1) * np.sqrt(2. / 1),
]

biases = [
  np.zeros((32, )),
  np.zeros((32, )),
  np.zeros((32, )),
  np.zeros((1, )),
]

weight_layers = [WeightLayer(weight) for weight in weights]
bias_layers = [BiasLayer(bias) for bias in biases]

dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

for i in range(1000000):
  for data in dataset:
    x_layer = InputLayer(np.array(data[0: 2]))
    for weight_layer, bias_layer in zip(weight_layers, bias_layers):
      x_layer = generate_affine_generator(x_layer, weight_layer, bias_layer)
      x_layer = SigmoidLayer(x_layer)
    mse_layer = MSELayer(x_layer, data[2])
    mse_layer.propagate()
    mse_layer.backpropagate(1, 0.001)
  if i % 1000 == 0:
    print(mse_layer.forward_res)

for data in dataset:
  x_layer = InputLayer(np.array(data[0: 2]))
  for weight_layer, bias_layer in zip(weight_layers, bias_layers):
    x_layer = generate_affine_generator(x_layer, weight_layer, bias_layer)
    x_layer = SigmoidLayer(x_layer)
  x_layer.propagate()
  print(x_layer.forward_res)
