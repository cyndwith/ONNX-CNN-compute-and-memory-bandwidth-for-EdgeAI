import onnx
import onnx.shape_inference
import numpy as np
from collections import defaultdict

# Dictionary to store initializer tensors and their shapes
initializer_dict = {}

def print_model_compute(model_path):
    model = onnx.load(model_path)
    # Perform shape inference on the model
    inferred_model = onnx.shape_inference.infer_shapes(model)

    graph = inferred_model.graph

    for initializer in model.graph.initializer:
        initializer_dict[initializer.name] = tuple(d for d in initializer.dims)

    total_flops = 0
    model_summary = defaultdict(int)
    print("\nLayer Dimensions:")
    for i, node in enumerate(graph.node):
        op_type = node.op_type
        output_name = node.output[0]
        model_summary[op_type] += 1
        if op_type == 'Conv':
            input_name = node.input[0]
            input_weight = node.input[1]
            input_shape = get_shape(graph, input_name)
            weight_shape = get_shape(graph, input_weight)
            output_shape = get_shape(graph, output_name)
            # print(node.attribute)
            # kernel_shape = node.attribute[2].ints
            for attribute in node.attribute:
              if attribute.name == 'kernel_shape':
                kernel_shape = attribute.ints
              if attribute.name == 'strides':
                stride_shape = attribute.ints
            # input dimensions
            C_in = input_shape[1]
            # output dimensions
            C_out = output_shape[1]
            H_out = output_shape[2]
            W_out = output_shape[3]
            # kernel dimensions
            H_k = kernel_shape[0]
            W_k = kernel_shape[1]
            # stride
            stride = stride_shape[0]
            # compute flops
            flops = C_in * H_out * W_out  * C_out * H_k * W_k / (stride * stride)
            total_flops += flops
            print(f"{i} Conv Layer: Input Shape: {input_shape}, weight: {weight_shape}, Output Shape: {output_shape}, flops: {flops}")

        elif op_type == 'Gemm':
            input_name = node.input[0]
            input_weight = node.input[1]
            input_shape = get_shape(graph, input_name)
            weight_shape = get_shape(graph, input_weight)
            output_shape = get_shape(graph, output_name)
            N_in = input_shape[1]
            H_w, W_w = weight_shape[0], weight_shape[1]
            flops = N_in * H_w * W_w
            total_flops += flops
            print(f"{i} Gemm {input_name} Layer: Input Shape: {input_shape}, weight: {weight_shape}, Output Shape: {output_shape}, flops: {flops}")

        elif op_type == 'AveragePool':
            input_name = node.input[0]
            input_shape = get_shape(graph, input_name)
            output_shape = get_shape(graph, output_name)
            kernel_shape = node.attribute[1].ints
            C_in = input_shape[1]
            H_out = output_shape[2]
            W_out = output_shape[3]
            H_k = kernel_shape[0]
            W_k = kernel_shape[1]
            flops = C_in * H_out * W_out * H_k * W_k
            total_flops += flops
            print(f"{i} AveragePool {input_name} Layer: Input Shape: {input_shape}, Output Shape: {output_shape}, flops: {flops}")

        elif op_type == 'GlobalAveragePool':
            input_name = node.input[0]
            input_shape = get_shape(graph, input_name)
            output_shape = get_shape(graph, output_name)
            C_in = input_shape[1]
            H_in = input_shape[2]
            W_in = input_shape[3]
            H_out = output_shape[2]
            W_out = output_shape[3]
            flops = C_in * H_out * W_out * H_in * W_in
            total_flops += flops
            print(f"{i} GlobalAveragePool {input_name} Layer: Input Shape: {input_shape}, Output Shape: {output_shape}, flops: {flops}")

        elif op_type == 'Add':
            input1_name = node.input[0]
            input2_name = node.input[1]
            input1_shape = get_shape(graph, input1_name)
            input2_shape = get_shape(graph, input2_name)
            output_shape = get_shape(graph, output_name)
            C = input_shape[1]
            H = input_shape[2]
            W = input_shape[3]
            flops = C * H * W
            total_flops += flops
            print(f"{i} Add Layer: Input1 Shape: {input1_shape}, Input2 Shape: {input2_shape}, Output Shape: {output_shape}, flops: {flops}")

        elif op_type == 'MatMul':
            input1_name = node.input[0]
            input2_name = node.input[1]
            input1_shape = get_shape(graph, input1_name)
            input2_shape = get_shape(graph, input2_name)
            output_shape = get_shape(graph, output_name)
            H_in, W_in = input_shape[0], input_shape[1]
            H_out, W_out = output_shape[0], output_shape[1]
            flops = H_in * W_in * H_out * W_out
            total_flops += flops
            print(f"{i} MatMul Layer: Input1 Shape: {input1_shape}, Input2 Shape: {input2_shape}, Output Shape: {output_shape}, flops: {flops}")
    # print model summary
    total_operations = 0
    print("model summary: {}".format(model_path))
    for op_type, op_num in model_summary.items():
      print("{}: {}".format(op_type, op_num))
      total_operations += op_num
    print("total operations: {}".format(total_operations))
    return total_flops

def get_shape(graph, name):
    for input_param in graph.input:
        if input_param.name == name:
            shape = [dim.dim_value for dim in input_param.type.tensor_type.shape.dim]
            # print('input_param:', input_param)
            return shape
    for value_info in graph.value_info:
        if value_info.name == name:
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            # print('value_info:', value_info)
            if len(shape) != 0:
              return shape
    for output_param in graph.output:
        if output_param.name == name:
            shape = [dim.dim_value for dim in output_param.type.tensor_type.shape.dim]
            # print('output_param:', output_param)
            return shape

    if name in initializer_dict:
        weight = initializer_dict[name]
        # print('initializer_dict:', weight)
        return weight

    return None

# Usage example
# # Path to the ONNX model file
# model_path = workspace + '/mobilenetv2-7.onnx'

# # Print input parameters and layer dimensions
# print_model_compute(model_path)
