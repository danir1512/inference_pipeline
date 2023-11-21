# inference_pipeline
This repository contains the project used for my Master thesis, Maritime Object Detection with Deep Learning for USV

A documentation with instruction has to how to install the program will follow in the future. This doocument page still needs to be finish.

The current filles present reprent the main files whihc havve the deepstream app with the ROS wrapper.

- `single_stream_class` represents the inference pipeline.
- `single_stream` represent the node which runs the inference pipeline
- `single_stream`sub detection represents an subcriber node that ssubcribe to the topic where the information is being publised and prints the message.

Most of these files where adapted from demo files that NVIDIA gave along side examples of how to use DeepStream. 
