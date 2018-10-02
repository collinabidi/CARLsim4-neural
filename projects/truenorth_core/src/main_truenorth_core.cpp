/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

// include CARLsim user interface
#include <carlsim.h>

// include stopwatch for timing
#include <stopwatch.h>

// include OPENCV for image loading/processing
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// include MNIST_READ to read in data properly
#include <mnist_reader.h>

#include <iostream>

int main() {
	// keep track of execution time
	Stopwatch watch;

	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 42;
	CARLsim sim("truenorth core simulation", GPU_MODE, USER, numGPUs, randSeed);

	// configure the network
	// set up a COBA three-layer network with full connectivity
	Grid3D gridIn(28, 28, 1); // input axons are on a 28 x 28 x 1 grid
	Grid3D gridHidden1(16, 16, 1); // hidden layer 1 neurons are on a 16 x 16 x 1 grid
	Grid3D gridHidden2(16, 16, 1); // hidden layer 2 neurons are on a 16 x 16 x 1 grid
	Grid3D gridOut(10, 1, 1); // output neurons are on a 10 x 1 grid

	// create groups
	int gin=sim.createSpikeGeneratorGroup("input", gridIn, EXCITATORY_NEURON);
	int ghide1 = sim.createGroup("hidden1", gridHidden1, EXCITATORY_NEURON);
	int ghide2 = sim.createGroup("hidden2", gridHidden2, EXCITATORY_NEURON);
	int gout=sim.createGroup("output", gridOut, EXCITATORY_NEURON);

	// set group parameters
	// membrane potential ->					v (mV)
	// recovery variable  ->					u
	// sum of all synaptic currents ->			I = i_synaptic + I_external
	// rate constant of u ->					a
	// sensitivity of u to fluctuations in v -> b
	//
	// change in recovery variable		du/dt = a(bv - u)
	// change in membrane potential		dv/dt = 0.04v^2 + 5v + 140 - u + I

	// currently set to regular spiking neurons
	sim.setNeuronParameters(ghide1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(ghide2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gout, 0.02f, 0.2f, -65.0f, 8.0f);

	// set connection weight/neuron parameters
	double low_w_in_hidden = -1;
	double high_w_in_hidden = 1;
	double low_w_hidden_out = -1;
	double high_w_hidden_out = 1;

	int in_hidden1 = sim.connect(gin, ghide1, "full", RangeWeight(low_w_in_hidden, high_w_in_hidden), 0.5f, RangeDelay(1), 
			RadiusRF(3, 3, 1), SYN_FIXED, 1.5f, 0.5f);
	int in_hidden2 = sim.connect(ghide1, ghide2, "full", RangeWeight(low_w_in_hidden, high_w_in_hidden), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_FIXED, 1.5f, 0.5f);
	int hidden_out = sim.connect(ghide2, gout, "full", RangeWeight(low_w_hidden_out, high_w_hidden_out), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_FIXED, 1.5f, 0.5f);
	sim.setConductances(true);

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	// set some monitors
	sim.setSpikeMonitor(gin,"DEFAULT");
	sim.setSpikeMonitor(ghide1, "DEFAULT");
	sim.setSpikeMonitor(ghide2, "DEFAULT");
	sim.setSpikeMonitor(gout,"DEFAULT");
	sim.setConnectionMonitor(gin,ghide1,"DEFAULT");
	sim.setConnectionMonitor(ghide1, ghide2, "DEFAULT");
	sim.setConnectionMonitor(ghide2, gout, "DEFAULT");

	//setup some baseline input
	PoissonRate in(gridIn.N, true);

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

	// load and format images from MNIST dataset
	Mat train_data_mat, train_label_mat;
	Mat test_data_mat, test_label_mat;

	loadMNIST("C:\\mnist\\data\\train-images.idx3-ubyte", "C:\\mnist\\data\\train-labels.idx1-ubyte", train_data_mat, train_label_mat);
	loadMNIST("C:\\mnist\\data\\t10k-images.idx3-ubyte", "C:\\mnist\\data\\t10k-labels.idx1-ubyte", test_data_mat, test_label_mat);

	train_data_mat.convertTo(train_data_mat, CV_32F);
	train_label_mat.convertTo(train_label_mat, CV_32F);
	test_data_mat.convertTo(test_data_mat, CV_32F);

	Mat train_image, train_label;
	// ************************* TRAIN IMAGES ***************************
	for (int i = 0; i<5; i++) {
		// load image from MNIST dataset
		train_image = train_data_mat.row(i);
		train_label = train_label_mat.row(i);

		// vectorize input image for spiking input
		std::vector<float> array((float*)train_image.data, (float*)train_image.data + train_image.rows * train_image.cols);

		// set input spikes based on input image
		in.setRates(array);
		sim.setSpikeRate(gin, &in);

		// run network for 150 ms for each image
		sim.runNetwork(0, 150);
	}

	// print stopwatch summary
	watch.stop();
	
	return 0;
}
