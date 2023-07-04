import onnx
import numpy as np
import onnxruntime as ort

def print_intermediate_results(model_path, input_data):
    model = onnx.load(model_path)
    graph = model.graph
    # collect intermediate output names
    output_names = []
    for i, item in enumerate(graph.node):
      output_names.append(item.output[0])
    # add intermediate outputs to onnx model
    shape_info = onnx.shape_inference.infer_shapes(model)
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for value_info in shape_info.graph.value_info:
      if value_info.name in output_names:
        value_info_protos.append(value_info)
    graph.output.extend(value_info_protos)
    # Perform shape inference on the model
    inferred_model = onnx.shape_inference.infer_shapes(model)

    graph = inferred_model.graph

    # save the modified onnx model
    modified_model_path = "modified_model.onnx"
    onnx.save(model, modified_model_path)
    # modified temp model
    model = onnx.load(modified_model_path)
    graph = model.graph
    # Collect intermediate results
    intermediate_results = {}
    graph_output_names = [output.name for output in graph.output]
    for node in graph.node:
        # Check if the node output is an intermediate result
        # Run inference for the node and retrieve the outputs
        session = ort.InferenceSession(modified_model_path)
        input_name = session.get_inputs()[0].name
        output_names = [output for output in graph_output_names]
        outputs = session.run(output_names, {input_name: input_data})

        # Store the intermediate results
        for name, output in zip(output_names, outputs):
            intermediate_results[name] = output
    memory_bandwidth = 0
    index = 0
    # Print intermediate results
    for name, result in intermediate_results.items():
        if 'batchnorm' in name or 'relu' in name:
          continue
        print(f'{index} Intermediate: {name}\tShape: {result.shape}\tSize: {result.size}')
        index += 1
        memory_bandwidth += result.size * 2
    print("memory / bandwidth: {}".format(memory_bandwidth))
    return memory_bandwidth

# # Usage example
# # Path to the ONNX model file
# model_path = workspace + '/test_model.onnx'
# input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)  # Adjust the shape and data type as per your model's input requirements
# # Print intermediate results
# print_intermediate_results(model_path, input_data)
