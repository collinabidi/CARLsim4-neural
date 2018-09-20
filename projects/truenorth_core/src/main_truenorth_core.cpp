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

	// load and format images from MNIST dataset
	Mat train_data_mat, train_label_mat;
	Mat test_data_mat, test_label_mat;

	loadMNIST("C:\\mnist\\data\\train-images.idx3-ubyte", "C:\\mnist\\data\\train-labels.idx1-ubyte", train_data_mat, train_label_mat);
	loadMNIST("C:\mnist\data\t10k-images.idx3-ubyte", "C:\mnist\data\t10k-labels.idx1-ubyte", test_data_mat, test_label_mat);

	train_data_mat.convertTo(train_data_mat, CV_32F);
	train_label_mat.convertTo(train_label_mat, CV_32F);
	test_data_mat.convertTo(test_data_mat, CV_32F);

	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 42;
	CARLsim sim("truenorth core simulation", GPU_MODE, USER, numGPUs, randSeed);

	// configure the network
	// set up a COBA three-layer network with full connectivity
	Grid3D gridIn(16, 16, 1); // input axons are on a 28 x 28 x 1 grid
	Grid3D gridHidden(10, 10, 1); // hidden layer neurons are on a 20 x 20 x 1 grid
	Grid3D gridOut(10, 1, 1); // output neurons are on a 10 x 1 grid

	// create groups
	int gin=sim.createSpikeGeneratorGroup("input", gridIn, EXCITATORY_NEURON);
	int ghide = sim.createGroup("hidden", gridHidden, EXCITATORY_NEURON);
	int gout=sim.createGroup("output", gridOut, EXCITATORY_NEURON);

	// set group parameters
	sim.setNeuronParameters(ghide, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gout, 0.02f, 0.2f, -65.0f, 8.0f);

	// set connections
	sim.connect(gin, ghide, "full", RangeWeight(0.05), 1.0f, RangeDelay(1), RadiusRF(3,3,1));
	sim.connect(ghide, gout, "full", RangeWeight(0.05), 1.0f, RangeDelay(1), RadiusRF(3, 3, 1));
	sim.setConductances(true);

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	// set some monitors
	sim.setSpikeMonitor(gin,"DEFAULT");
	sim.setSpikeMonitor(ghide, "DEFAULT");
	sim.setSpikeMonitor(gout,"DEFAULT");
	sim.setConnectionMonitor(gin,ghide,"DEFAULT");
	sim.setConnectionMonitor(ghide, gout, "DEFAULT");

	// load in input image
	//cv::Mat imageFlip = train_data_mat.row(0);
	Mat mat;
	//cv::flip(imageFlip, image, 1);
	mat = cv::imread("../../projects/truenorth_core/input_data/two.bmp", cv::IMREAD_GRAYSCALE);	// Read the file, convert to greyscale
	std::vector<float> array;

	if (!mat.data)	// Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	// convert image to 2D vector that will act as input
	else
	{
		if (mat.isContinuous()) {
			array.assign((float*)mat.datastart, (float*)mat.dataend);
		}
		else {
			for (int i = 0; i < mat.rows; ++i) {
				array.insert(array.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols);
			}
		}
	}

	//setup some baseline input
	PoissonRate in(gridIn.N, true);
	in.setRates(array);
	sim.setSpikeRate(gin, &in);


	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	for (int i = 0; i<20; i++) {
		sim.runNetwork(0, 120);
	}

	// print stopwatch summary
	watch.stop();
	
	return 0;
}
