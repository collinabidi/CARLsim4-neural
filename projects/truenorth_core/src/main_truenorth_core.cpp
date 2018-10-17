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
	Grid3D core1_in(16, 16, 1);		// input axons are on a		256 x 1 x 1 grid
	Grid3D core1_out(16, 16, 1);	// output neurons are on a	256 x 1 x 1 grid
	Grid3D core2_in(16, 16, 1);		// input axons are on a		256 x 1 x 1 grid
	Grid3D core2_out(16, 16, 1);	// output neurons are on a	256 x 1 x 1 grid
	Grid3D core3_in(16, 16, 1);		// input axons are on a		256 x 1 x 1 grid
	Grid3D core3_out(16, 16, 1);	// output neurons are on a	256 x 1 x 1 grid
	Grid3D core4_in(16, 16, 1);		// input axons are on a		256 x 1 x 1 grid
	Grid3D core4_out(16, 16, 1);	// output neurons are on a	256 x 1 x 1 grid
	Grid3D result(10, 1, 1);		// output vector (digits 0-9)

	// create groups
	int gin_1=sim.createSpikeGeneratorGroup("core1_input", core1_in, EXCITATORY_NEURON);
	int gout_1=sim.createGroup("core1_output", core1_out, EXCITATORY_NEURON);
	int gin_2 = sim.createSpikeGeneratorGroup("core2_input", core2_in, EXCITATORY_NEURON);
	int gout_2 = sim.createGroup("core2_output", core2_out, EXCITATORY_NEURON);
	int gin_3 = sim.createSpikeGeneratorGroup("core3_input", core3_in, EXCITATORY_NEURON);
	int gout_3 = sim.createGroup("core3_output", core3_out, EXCITATORY_NEURON);
	int gin_4 = sim.createSpikeGeneratorGroup("core4_input", core4_in, EXCITATORY_NEURON);
	int gout_4 = sim.createGroup("core4_output", core4_out, EXCITATORY_NEURON);
	int gresult = sim.createGroup("result_output", result, EXCITATORY_NEURON);

	// currently set to regular spiking neurons
	sim.setNeuronParameters(gout_1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gout_2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gout_3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gout_4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gresult, 0.02f, 0.2f, -65.0f, 8.0f);

	// set connection weight/neuron parameters
	int in_out_1 = sim.connect(gin_1, gout_1, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int in_out_2 = sim.connect(gin_2, gout_2, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int in_out_3 = sim.connect(gin_3, gout_3, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int in_out_4 = sim.connect(gin_4, gout_4, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int out_result_1 = sim.connect(gout_1, gresult, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int out_result_2 = sim.connect(gout_2, gresult, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int out_result_3 = sim.connect(gout_3, gresult, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	int out_result_4 = sim.connect(gout_4, gresult, "full", RangeWeight(30.0), 0.5f, RangeDelay(1),
		RadiusRF(3, 3, 1), SYN_PLASTIC, 1.5f, 0.5f);
	sim.setConductances(true);
	

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	// set some monitors
	sim.setSpikeMonitor(gin_1,"DEFAULT");
	sim.setSpikeMonitor(gin_2, "DEFAULT");
	sim.setSpikeMonitor(gin_3, "DEFAULT");
	sim.setSpikeMonitor(gin_4, "DEFAULT");
	sim.setSpikeMonitor(gout_1,"DEFAULT");
	sim.setSpikeMonitor(gout_2, "DEFAULT");
	sim.setSpikeMonitor(gout_3, "DEFAULT");
	sim.setSpikeMonitor(gout_4, "DEFAULT");
	sim.setSpikeMonitor(gresult, "DEFAULT");

	//setup some baseline input
	PoissonRate in_1(core1_in.N, true);
	PoissonRate in_2(core2_in.N, true);
	PoissonRate in_3(core3_in.N, true);
	PoissonRate in_4(core4_in.N, true);

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
		std::vector<float> array_1;
		std::vector<float> array_2;
		std::vector<float> array_3;
		std::vector<float> array_4;
		// vectorize inputs for each of the four cores
		for (int j = 0; j < 16; j++)
		{
			array_1.insert(array_1.end(), array.begin() + j * 28, array.begin() + (j * 28 + 16));
			array_2.insert(array_2.end(), array.begin() + (j * 28 + 4), array.begin() + (j * 28 + 20));
			array_3.insert(array_3.end(), array.begin() + (j + 12) * 28, array.begin() + ((j + 12) * 28 + 16));
			array_4.insert(array_4.end(), array.begin() + ((j + 12) * 28 + 4), array.begin() + ((j + 12) * 28 + 20));
		}
		// set input spikes based on input image
		cout << array_1.size() << endl;
		cout << array_2.size() << endl;
		cout << array_3.size() << endl;
		cout << array_4.size() << endl;
		in_1.setRates(array_1);
		sim.setSpikeRate(gin_1, &in_1);
		in_2.setRates(array_2);
		sim.setSpikeRate(gin_2, &in_2);
		in_3.setRates(array_3);
		sim.setSpikeRate(gin_3, &in_3);
		in_4.setRates(array_4);
		sim.setSpikeRate(gin_4, &in_4);

		// run network for 150 ms for each image
		sim.runNetwork(0, 500);
	}

	// print stopwatch summary
	watch.stop();
	
	return 0;
}
