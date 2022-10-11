import numpy as np
from openvino.runtime import Core

inp = np.load('input.npy')
inp = np.expand_dims(inp, axis=0)

core = Core()
net = core.read_model('midas.xml')
compiled_model = core.compile_model(net, 'CPU')

results = compiled_model.infer_new_request({0: inp})
out = next(iter(results.values()))

print(out.shape)

ref = np.load('ref.npy')
print(ref.shape)
maxdiff = np.max(np.abs(ref - out))

print('Reference range: [{}, {}]'.format(np.min(ref), np.max(ref)))
print('Maximal difference:', maxdiff)