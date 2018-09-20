#ifndef _MNIST_READER_
#define _MNIST_READER_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PIXELS_IN_IMAGE 28*28

#define ENABLE_TRAIN 1

int reverseInt(int i);
int loadMNIST(const string pic_filename, const string label_filename, Mat& training_data, Mat& label_data);

/*
int main(int argc, char* argv[])
{
	try
	{
		Mat train_data_mat, train_label_mat;
		Mat test_data_mat, test_label_mat;

		loadMNIST("F:\\Datasets\\mnist\\train-images.idx3-ubyte", "F:\\Datasets\\mnist\\train-labels.idx1-ubyte", train_data_mat, train_label_mat);
		loadMNIST("F:\\Datasets\\mnist\\t10k-images.idx3-ubyte", "F:\\Datasets\\mnist\\t10k-labels.idx1-ubyte", test_data_mat, test_label_mat);

		train_data_mat.convertTo(train_data_mat, CV_32FC1);
		train_label_mat.convertTo(train_label_mat, CV_32SC1);
		test_data_mat.convertTo(test_data_mat, CV_32FC1);

		Ptr<ml::KNearest> knn;
		knn = ml::KNearest::create();
		knn->setDefaultK(10);
		knn->train(train_data_mat, ml::SampleTypes::ROW_SAMPLE, train_label_mat);
		int correct_count = 0;
		for (int idx = 0; idx < test_label_mat.rows; idx++) {
			Mat result_mat;
			float response = knn->findNearest(test_data_mat.row(idx), knn->getDefaultK(), result_mat);
			if (test_label_mat.at<uchar>(idx, 0) == (uchar)response) {
				correct_count++;
			}
		}

		double correct_ratio = (double)correct_count / (double)test_label_mat.rows;
		cout << correct_ratio << endl;

	}
	catch (const Exception& ex)
	{
		cout << "Error: " << ex.what() << endl;
	}

	cin.get();

	return 0;
}
*/

int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int loadMNIST(const string pic_filename, const string label_filename, Mat& training_data, Mat& label_data) {
	std::ifstream pic_file(pic_filename, std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::binary);

	if (pic_file.is_open() && label_file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		label_file.read((char*)&magic_number, sizeof(magic_number));
		pic_file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		label_file.read((char*)&number_of_images, sizeof(number_of_images));
		pic_file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);

		pic_file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		pic_file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		int n_stride = n_cols * n_rows;
		training_data = Mat(number_of_images, n_stride, CV_8U);
		label_data = Mat(number_of_images, 1, CV_8U);

		for (int i = 0; i < number_of_images; ++i) {
			//		for (int i = 0; i < 5000; ++i) {

			unsigned char data_tmp[PIXELS_IN_IMAGE];
			pic_file.read((char*)data_tmp, sizeof(unsigned char) * n_stride);
			Mat row_image(1, n_stride, CV_8U, data_tmp);
			row_image.row(0).copyTo(training_data.row(i));

			char label = 0;
			label_file.read((char*)&label, sizeof(label));
			label_data.at<uchar>(i, 0) = label;
		}
	}
	else {
		return 1;
	}
	return 0;
}

#endif // _MNIST_READER_