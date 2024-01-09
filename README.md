# ONNX-CNN-Compute-and-Memory-bandwidth-for-Edge-AI
ONNX tools to estimate the compute and memory bandwidth of CNN models.

# Models

ONNX models used for the evaluation are available at: https://github.com/onnx/models?tab=readme-ov-file#image_classification

# Network Architectures vs Computer / Bandwidth

we estimate the compute complexity of the CNN through FLOPs, which can better estimate operations like elementwise
add/multiply, average pooling which are found more frequently in many of these modern architectures. 

<img width="1013" alt="Screen Shot 2024-01-08 at 9 48 44 PM" src="https://github.com/cyndwith/ONNX-CNN-compute-and-memory-bandwidth-for-EdgeAI/assets/11755434/a6033af2-7c48-4d77-84b5-0bd3c4679207">

These estimates are made per model inference, by calculating the intermediate activation and weights that need to be moved in and out of local and system memory. 

<img width="1034" alt="Screen Shot 2024-01-08 at 9 49 03 PM" src="https://github.com/cyndwith/ONNX-CNN-compute-and-memory-bandwidth-for-EdgeAI/assets/11755434/9f0528d1-9463-4de8-9d2a-6c5bc98ab9d1">

# Computer (FLOPs) vs Memory Bandwidth (/Inference)
A comparison of the compute vs memory bandwidth shown in Fig. below, shows that mobilenet networks designed for mobile application tend to
be leaner and deeper, requiring higher bandwidth, might not be optimal architecture for the edge AI application. SqueezeNet shows
superior performance in terms of both compute and memory bandwidth tradeoff, as it was designed for smaller size and minimal memory bandwidth. 

<img width="1044" alt="Screen Shot 2024-01-08 at 9 49 58 PM" src="https://github.com/cyndwith/ONNX-CNN-compute-and-memory-bandwidth-for-EdgeAI/assets/11755434/7ba9bb8c-f51c-4305-81fb-04cbe03bcff6">

# Reference
[1] Dwith Chenna, Evolution of Convolutional Neural Network(CNN): Compute vs Memory bandwidth for Edge AI, 
IEEE FeedForward Magazine 2(3), 2023, pp. 3-13.

Paper link: https://arxiv.org/pdf/2311.12816.pdf
