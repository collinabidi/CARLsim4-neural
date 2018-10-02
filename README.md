#CARLsim4 Neural Network Simulations

This project uses CARLsim 4, developed by Cognitive Anteater Robotics Laboratory (http://www.socsci.uci.edu/~jkrichma/CARLsim/)

CARLsim is an efficient, easy-to-use, GPU-accelerated library for simulating large-scale spiking neural network (SNN) models with a high degree of biological detail. CARLsim allows execution of networks of Izhikevich spiking neurons with realistic synaptic dynamics on both generic x86 CPUs and standard off-the-shelf GPUs. The simulator provides a PyNN-like programming interface in C/C++, which allows for details and parameters to be specified at the synapse, neuron, and network level.

This project aims to simulate different neurmorphic chip designs by mapping architectures to an equivalent SNN representation that achieves performance similar to that of the real-world hardware in order to bridge the gap between CNNs and SNNs.

#TrueNorth

The TrueNorth branch is a simulation of the IBM TrueNorth chip, a neuromorphic CPU designed to simulate mammalian brain image processing functions.
Currently working on implementing Deep Learning techniques to classify handwritten digits using the MNIST Database.


#Citations/Links

Chou*, T.-S., Kashyap*, H.J., Xing, J., Listopad, S., Rounds, E.L., Beyeler, M., Dutt, N., and Krichmar, J.L. (2018). "CARLsim 4: An Open Source Library for Large Scale, Biologically Detailed Spiking Neural Network Simulation using Heterogeneous Clusters." In Proceedings of IEEE International Joint Conference on Neural Networks (IJCNN), pp. 1158-1165. (*co-first authors)
URL: (https://www.socsci.uci.edu/~jkrichma/Chou-Kashyap-CARLsim4-IJCNN2018.pdf)

OpenCV
(https://opencv.org/)

CARLsim GitHub
(https://github.com/UCI-CARL/CARLsim4)

IBM TrueNorth
J. Sawada et al., "TrueNorth Ecosystem for Brain-Inspired Computing: Scalable Systems, Software, and Applications," SC '16: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, Salt Lake City, UT, 2016, pp. 130-141.
doi: 10.1109/SC.2016.11
URL: (http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7877010&isnumber=7876994)
