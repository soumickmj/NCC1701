#After installing with pip, replace inside the site packages, teh following from this git:-
#https://github.com/afqueiruga/pytorchviz/tree/master/torchviz

from torchviz import make_dot, make_dot_from_trace, trace_and_dot
import torch

def generateGraph(input, params, output_path, output_format='pdf', additional_vars=None):
    #input: Loss, model's output whatever
    #params can be: model.named_parameters()
    #additional_vars: should be supplied as a list of tuples. For example, [('x', x), ('data', y)]
    if additional_vars is not None:
        params = list(params) + additional_vars
    graph = make_dot(loss, params=dict(params))
    graph.format = output_format
    graph.render(output_path)

def generateGraphFromTrace(model, model_input, output_path, output_format='pdf'):
    #with torch.onnx.set_training(model, False):
    #    trace, _ = torch.jit.get_trace_graph(model, args=(model_input,))
    #graph = make_dot_from_trace(trace)
    graph = trace_and_dot(model, model_input)
    graph.format = output_format
    graph.render(output_path)