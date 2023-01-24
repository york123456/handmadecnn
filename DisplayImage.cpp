#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<iostream>
#include <chrono>
#include <thread>
#include <future>

#define w 400
using namespace cv;

double h = 1e-7;
double learning_rate = 1e-2;
int batch_size = 1;
const int n =3000;

double b_1 = 0.9;
double b_2 = 0.999;

const int size_of_X = 1;

double* X = (double*)malloc(sizeof(double) * size_of_X);
double* X2 = (double*)malloc(sizeof(double) * size_of_X);
double* Y = (double*)malloc(sizeof(double) * size_of_X);


double* A = (double*)malloc(sizeof(double) * n);
double* mt_1 = (double*)malloc(sizeof(double) * n);
double* vt_1 = (double*)malloc(sizeof(double) * n);
double* mt = (double*)malloc(sizeof(double) * n);
double* vt = (double*)malloc(sizeof(double) * n);
double* _m = (double*)malloc(sizeof(double) * n);
double* _v = (double*)malloc(sizeof(double) * n);
double* DROPOUT = (double*)malloc(sizeof(double) * n);


int data=20;  //40張

const int img_w = 8 * 8;
double x_input[img_w] = { 0 };


double f(double x) {
	//printf("%f  ", 62 * pow(x, 2) - x + 99);
	return -18 * pow(x, 5) + 50 * pow(x, 4) - x + 30;
}

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double relu(double x) {
	if (x > 0)return x;
	if (x <= 0)return x*0.01;
}

double Tanh(double x) {
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

//double a_hidden_B_arr[hidden_layer_B];
int Dense_relu(const int hidden_layer_A,const int hidden_layer_B,double hidden_layer_A_arr[],double hidden_B_arr[],double AK[],int temp){
	
	int i,j;
	for (i = 0; i < hidden_layer_B; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_layer_A; j++) {
			x_sum += AK[temp + j] * hidden_layer_A_arr[j];
		}
		x_sum += AK[temp +  j];
		hidden_B_arr[i] = relu(x_sum);

		temp = temp + hidden_layer_A + 2;
		
	}
	return temp;

}


int Dense_sigmoid(const int hidden_layer_A,const int hidden_layer_B,double hidden_layer_A_arr[],double hidden_B_arr[],double AK[],int temp){
	
	int i,j;
	for (i = 0; i < hidden_layer_B; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_layer_A; j++) {
			x_sum += AK[temp + j] * hidden_layer_A_arr[j];
		}
		x_sum += AK[temp +  j];
		hidden_B_arr[i] = sigmoid(x_sum);

		temp = temp + hidden_layer_A + 2;
		
	}
	return temp;

}


int input_layer_relu(const int hidden_layer_A,double hidden_layer_A_arr[],double AK[],int temp){
	int i,j;
	for (i = 0; i < hidden_layer_A; i++) {
		hidden_layer_A_arr[i] = relu(AK[i * 2 + 0] * x_input[i] + AK[i * 2 + 1]);

	}
	temp = i*2+1; 
	return temp;    
	
}

int output_layer_sigmoid(const int hidden_layer_A,const int hidden_layer_B,double hidden_layer_A_arr[],double hidden_B_arr[],double AK[],int temp){

	int i,j;
	for (i = 0; i < hidden_layer_B; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_layer_A; j++) {
			x_sum += AK[temp + j] * hidden_layer_A_arr[j];
		}
		x_sum += AK[temp + j ];

		hidden_B_arr[i] = sigmoid(x_sum);
		temp = temp + hidden_layer_A + 2;
	}
	return temp;
		
}

int conv(const int kernel_size,const int RRR,const int hidden_layer_A,const int hidden_layer_B,double hidden_layer_A_arr[],double hidden_B_arr[],double AK[],int temp){
	int i,j;
	int rrr;
	int counter=0;
	for(rrr=0;rrr<RRR;rrr++){
	
		double kernel[kernel_size];
		for(i=0;i<kernel_size;i++){
			kernel[i]=AK[temp+i];
		}
		temp=temp+i+2;
		int k,l;
		
		for(i=0;i<sqrt(hidden_layer_A)-sqrt(kernel_size)+1;i++){
			for(j=0;j<sqrt(hidden_layer_A)-sqrt(kernel_size)+1;j++){
				double sum = 0;
				for(k=0;k<sqrt(kernel_size);k++){
					for(l=0;l<sqrt(kernel_size);l++){
						sum+=hidden_layer_A_arr[int(i+j*sqrt(kernel_size)+k+l*sqrt(kernel_size))]*kernel[int(k+l*sqrt(kernel_size))];
					}			
				}	
				hidden_B_arr[counter]= sum;
				counter+=1;
				
			}
			
		}
		
	}
	
	return temp;
}

void maxpooling(const int maxpooling_size,const int hidden_layer_A,const int pool_layer,double hidden_layer_A_arr[],double a_pool_layer[]){
	int i,j,k,l;
	for(i=0;i<sqrt(hidden_layer_A)/sqrt(maxpooling_size);i++){
		for(j=0;j<sqrt(hidden_layer_A)/sqrt(maxpooling_size);j++){
			
			double max=0;
			for(k=0;k<sqrt(maxpooling_size);k++){
				for(l=0;l<sqrt(maxpooling_size);l++){
					if(max<hidden_layer_A_arr[int((i+j*sqrt(maxpooling_size))*sqrt(maxpooling_size)+k+l*sqrt(maxpooling_size))])
					{
					max=hidden_layer_A_arr[int((i+j*sqrt(maxpooling_size))*sqrt(maxpooling_size)+k+l*sqrt(maxpooling_size))];
}				
				
				}
			}
			a_pool_layer[int(i+j*sqrt(maxpooling_size))]=max;
			
		
	
	}
	}
	
}



double F(double AK[], double x_input[]) {

	int i, j;
	int temp=0;

	//input layer
	double a_input[img_w];
	temp=input_layer_relu(img_w,a_input,AK,temp);
	
	//conv 3*3
	const int hidden_layer_A=img_w;
	const int kernel_size=3*3;
	const int RRR_2=1;
	const int conv_layer = (sqrt(hidden_layer_A)-sqrt(kernel_size)+1)*(sqrt(hidden_layer_A)-sqrt(kernel_size)+1)*RRR_2;
	double a_conv_layer[conv_layer];
	temp=conv(kernel_size,RRR_2,hidden_layer_A,conv_layer,a_input,a_conv_layer,AK,temp);

	//conv 2*2*5
	const int RRR=5;
	const int hidden_layer_A_2=conv_layer;
	const int kernel_size_2=2*2;
	const int conv_layer_2 = (sqrt(hidden_layer_A_2)-sqrt(kernel_size_2)+1)*(sqrt(hidden_layer_A_2)-sqrt(kernel_size_2)+1)*RRR;
	double a_conv_layer_2[conv_layer_2];
	temp=conv(kernel_size_2,RRR,hidden_layer_A_2,conv_layer_2,a_conv_layer,a_conv_layer_2,AK,temp);
	
	//maxpooling
	const int hidden_layer_A_3=conv_layer_2;
	const int maxpooling_size=3*3;
	const int pool_layer=(sqrt(hidden_layer_A_3)/sqrt(maxpooling_size))*(sqrt(hidden_layer_A_3)/sqrt(maxpooling_size));
	double a_pool_layer[pool_layer];
	maxpooling(maxpooling_size,hidden_layer_A_3,pool_layer,a_conv_layer_2,a_pool_layer);
	
	//dense 10
	const int hidden_2 = 10;
	double a_hidden_2[hidden_2];
	temp=Dense_relu(pool_layer,hidden_2,a_pool_layer,a_hidden_2,AK,temp);
	
	//dense 10
	const int hidden_3 = 10;
	double a_hidden_3[hidden_3];
	temp=Dense_relu(hidden_2,hidden_3,a_hidden_2,a_hidden_3,AK,temp);

	//output 1
	const int output = 1;
	double a_output[output];
	temp=output_layer_sigmoid(hidden_3,output,a_hidden_3,a_output,AK,temp);

	
	//printf("->%d",temp);
	return a_output[0];
}


double partialW(double A[], double  x_input[], double y, int N) {

	double A1[n] = { 0 };
	double A2[n] = { 0 };
	int i;
	for (i = 0; i < n; i++) {
		A1[i] = A[i];
		A2[i] = A[i];

	}
	A1[N] += h;

	//printf("\n");
	//printf("<%f--%f--%f>\n   ",  F(A1, x_input), pow(y - F(A1, x_input), 2), (pow(y - F(A1, x_input), 2) - pow(y - F(A2, x_input), 2)) / h);
	return (pow(y - F(A1, x_input), 2) - pow(y - F(A2, x_input), 2)) / h;


}


//Adam
double train(double A[], double x_input[], double y) {
	int i, j;
	for (j = 0; j < n; j++) {
		
		//Dropout
		if (DROPOUT[j] == 0){

		int arr[size_of_X] = { 0 };
		//printf("%f", size_of_X);
		for (i = 0; i < size_of_X; i++) {
			arr[i] = 0;
		}

		for (i = 0; i < batch_size; i++) {
			int index = rand() % size_of_X;
			while (arr[index] == 1) {
				index = rand() % size_of_X;

			}
			//printf("eWegerh");
			//printf("%d",index);

			arr[index] = 1;


			//x_input[0] = x;
			//x_input[1] = x2;

			double g = partialW(A, x_input, y, j);
			//printf("<%f>", g);

			mt[j] = b_1 * mt_1[j] + (1 - b_1) * g;
			vt[j] = b_2 * vt_1[j] + (1 - b_2) * pow(g, 2);
			_m[j] = mt[j] / (1 - b_1);
			_v[j] = vt[j] / (1 - b_2);

			vt_1[j] = vt[j];
			mt_1[j] = mt[j];
			
			if(A[j] - learning_rate * _m[j] / (sqrt(_v[j]) + 1e-8)<2 and A[j] - learning_rate * _m[j] / (sqrt(_v[j]) + 1e-8)>-2)
			 A[j] = A[j] - learning_rate * _m[j] / (sqrt(_v[j]) + 1e-8);

		}
		}
	}
	return 0;

}
int main() {
	
	srand(time(NULL));


	//Mat atom_image = Mat::zeros(w, w, CV_8UC3);

	/*Point p2;
	p2.x = 100;
	p2.y = 100;
	//画实心点
	circle(atom_image, p2, 3, Scalar(255, 0, 0), -1);
	*/
	int i, j;

	for (i = 0; i < n; i++) {
		A[i] = rand() % 200 / 100.0 - 1;
		mt_1[i] = 0;
		vt_1[i] = 0;
		mt[i] = 0;
		vt[i] = 0;
		_m[i] = 0;
		_v[i] = 0;
		DROPOUT[i] = 0;
	}

	/*
	for (i = 0; i < size_of_X; i++) {
		X[i] = i / 100.0 - 1;
		if (i > size_of_X / 2) { Y[i]=1; }
		else { Y[i]=0; }
	}*/

	double error = 0;
	double lerr = 999;

	for (i = 0; i < 500; i++) {
	auto start = std::chrono::system_clock::now();
		printf("!");

		for(j=0;j<n;j++)DROPOUT[j] = 0;


		for (j = 0; j < n * 0.85; j++) {
			int index = rand() % n;
			while (DROPOUT[index] == 1) {
				index = rand() % n;
			}
			DROPOUT[index] = 1;
		}
		
		printf("\n");
		int rrr = 0;
		error = 0;
		
		for (rrr = 0; rrr < 16; rrr++) {



			int kkk = 0;
			char str[] = "./0/";
			String str2 = std::to_string(rand() % data);
			char str3[] = ".png";
			String Str;
			Str.append(str);
			Str.append(str2);
			Str.append(str3);

			std::cout<< ".";
			fflush(stdout); 
			
			Mat img2 = imread(Str, 0);
			Mat img;
			resize(img2, img, Size(sqrt(img_w), sqrt(img_w)), INTER_LINEAR);

			int i, j;
			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img.at<uchar>(i, j) /  127.0-1;

				}
			}



			//imshow("test", img2);
			//waitKey(1);

			//printf("%f", A[2]);

			std::thread t1(train,A, x_input, 0);
			t1.join();
			//auto future = std::async(train, A, x_input, 0);
		        //int simple = future.get();
		        //std::cout<<simple;
			
			//train(A, x_input, 0);

			char strA[] = "./1/";
			String str2A = std::to_string(rand() % data);
			char str3A[] = ".png";
			String StrA;
			StrA.append(strA);
			StrA.append(str2A);
			StrA.append(str3A);

			std::cout<< ".";
			 fflush(stdout); 
			//printf(".");
			
			Mat img3 = imread(StrA, 0);
			Mat img4;
			resize(img3, img4, Size(sqrt(img_w), sqrt(img_w)), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img4.at<uchar>(i, j) /  127.0-1;

				}
			}



			//imshow("test", img3);
			//waitKey(1);
			//destroyAllWindows();
			//auto future_2 = std::async(train, A, x_input, 1);
		        //int simple_2 = future_2.get();
		        
			//train(A, x_input, 1);
			//waitKey(1);
			std::thread t2(train,A, x_input, 1);
			t2.join();
			
			//t1.join();
			//t2.join();
			/*
			if(i%10==0){
				for (j = 0; j < 200; j++) {
					Point p1;
					p1.x = j*2;
					p1.y =0.5*w-f(j/100.0-1);
					//画实心点5
					circle(atom_image, p1, 1, Scalar(255,255, 255), -1);
					Point p2;
					p2.x = j * 2;
					x_input[0] = j / 100.0 - 1;
					x_input[1] = 1.3*j / 100.0 - 1;
					p2.y = 0.5*w-F(A, x_input);
					//画实心点
					circle(atom_image, p2, 1, Scalar(255, 0, 255), -1);
				}
				//imshow("test", atom_image);
				//waitKey(1);
			}*/


			


			char strB[] = "./0/";
			String str2B = std::to_string(rand() % data);
			char str3B[] = ".png";
			String StrB;
			StrB.append(strB);
			StrB.append(str2B);
			StrB.append(str3B);

			std::cout<<  ".";
			 fflush(stdout); 
			//printf(".");

			Mat img5 = imread(StrB, 0);
			Mat img6;
			resize(img5, img6, Size(sqrt(img_w), sqrt(img_w)), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img6.at<uchar>(i, j) /  127.0-1;

				}
			}
			error += pow(0 - F(A, x_input), 2);


			//imshow("test", img5);
			//waitKey(1);



			//train(A, x_input, 0);
			std::thread t3(train,A, x_input, 0);
			t3.join();
			//auto future_3 = std::async(train, A, x_input, 0);
		        //int simple_3 = future_3.get();

			char strC[] = "./1/";
			String str2C = std::to_string(rand() % data);
			char str3C[] = ".png";
			String StrC;
			StrC.append(strC);
			StrC.append(str2C);
			StrC.append(str3C);

			std::cout<<  ".";
			 fflush(stdout); 
			//printf(".");

			Mat img7 = imread(StrC, 0);
			Mat img8;
			resize(img7, img8, Size(sqrt(img_w), sqrt(img_w)), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img8.at<uchar>(i, j) /  127.0-1;

				}
			}
			
			//imshow("test", img7);
			//waitKey(1);
			//train(A, x_input, 1);
			//auto future_4 = std::async(train, A, x_input, 1);
		        //int simple_4 = future_4.get();
			std::thread t4(train,A, x_input, 1);
			t4.join();

			error += pow(1 - F(A, x_input), 2);

		}
		printf("loss = %f \n", error);

		/*for (j = 0; j < n; j++) {
			printf("%f ", A[j]);
		}
		printf("\n");*/

		if (error < 1.5) {
			break;
			printf("!!!!!!!!!!!");
		}



		lerr = error;
		
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
    		std::cout << "Elapsed Time: " << elapsed_seconds.count() << " sec" << std::endl;
 

	}

	/*
	printf("\n"); printf("\n"); printf("\n");
	for (j = 0; j < n; j++) {
		printf("%f  ", A[j]);
	}
	printf("\n"); printf("\n"); printf("\n");
	*/

	while (true)
	{
		int input;
		printf("input image(0 is 0，1 is 1):");
		scanf("%d", &input);
		if (input == 0) {

			char strC[] = "./0/";
			String str2C = std::to_string(rand() % data);
			char str3C[] = ".png";
			String StrC;
			StrC.append(strC);
			StrC.append(str2C);
			StrC.append(str3C);

			//printf("%s", StrC);

			Mat img9 = imread(StrC, 0);

			Mat img10;
			resize(img9, img10, Size(sqrt(img_w), sqrt(img_w)), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img10.at<uchar>(i, j) /  127.0-1;

				}
			}
			int txt = 1;
			if (F(A, x_input) < 0.5)txt = 0;
			printf("predict value=%f  predict class=%d\n\n", F(A, x_input), txt);
			imshow("test", img9);
			waitKey(10);
		}
		if (input == 1) {

			char strC[] = "./1/";
			String str2C = std::to_string(rand() % data);
			char str3C[] = ".png";
			String StrC;
			StrC.append(strC);
			StrC.append(str2C);
			StrC.append(str3C);

			//printf("%s", StrC);

			Mat img11 = imread(StrC, 0);

			Mat img12;
			resize(img11, img12, Size(sqrt(img_w), sqrt(img_w)), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img12.at<uchar>(i, j) / 127.0-1;

				}
			}
			int txt = 1;
			if (F(A, x_input) < 0.5)txt = 0;
			printf("predict value=%f  predict class=%d\n\n", F(A, x_input), txt);
			imshow("test", img11);
			waitKey(10);
		}

	}

	//waitKey(0);
	//destroyAllWindows();
	destroyAllWindows();
	system("pause");
	return 0;
}
