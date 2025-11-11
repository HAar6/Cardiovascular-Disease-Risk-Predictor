import os
from tensorflow.keras.models import load_model
import tf2onnx

h5 = os.path.join('backend','binary_model.h5')
onnx = os.path.join('backend','binary_model.onnx')
print('H5 exists?', os.path.exists(h5))
model = load_model(h5)
print('Loaded Keras model, converting to ONNX...')
# Build a simple input signature for 13 features
import tensorflow as _tf
spec = (_tf.TensorSpec((None,13), dtype=model.inputs[0].dtype, name='input'),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx)
print('Saved ONNX to', onnx)
