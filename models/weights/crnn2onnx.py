import torch
import onnx
import crnn as crnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required = True, help = 'path to model')
# parser.add_argument('--output', required = True, help = 'path to output model')
opt = parser.parse_args()

# ===========================================================
model_path = opt.model
model = crnn.CRNN(32, 1, 40, 256)
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path, ).items()})

input_names = ['input_1']
output_names = ['output_1']
dummy_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, dummy_input, 'crnnCN.onnx',
        input_names = input_names,
        output_names = output_names)

print('================================================================')
model = onnx.load('crnnCN.onnx')
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

