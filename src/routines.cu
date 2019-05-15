#include <iostream>
#include <cmath>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <stdlib.h>

#include <string>
#include "H5Cpp.h"

#include <time.h>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace H5;


void output(double *h_result, string hh_file, string ss_file, int nElem_x, int nElem_t)
{
    cout << hh_file <<endl;
    const H5std_string FILE_NAME(ss_file);
    const H5std_string DATASET_NAME("FloatArray");
    H5File file( FILE_NAME, H5F_ACC_TRUNC );

    hsize_t dimsf[2];
    dimsf[0] = nElem_x;
    dimsf[1] = nElem_t;
    DataSpace dataspace(2, dimsf);

    IntType datatype(PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
    dataset.write(h_result, PredType::NATIVE_DOUBLE);
}


__device__ void func_cal(double &func_result, double time_cal, double x0_cal)
{
    func_result = cos(time_cal*time_cal*1e5);
}

__device__ void rk4_tryStep(double &time_fin, double &x0_fin, double time_init, double x0_init, double step_time)
{
    double t0, t1, t2, t3;
    double u0, u1, u2, u3;
    double f0, f1, f2, f3;

    t0 = time_init;
    u0 = x0_init;
    func_cal(f0, t0, u0);

    t1 = t0 + step_time / 2.0;
    u1 = u0 + step_time * f0 / 2.0;
    func_cal(f1, t1, u1);

    t2 = t0 + step_time / 2.0;
    u2 = u0 + step_time * f1 / 2.0;
    func_cal(f2, t2, u2);

    t3 = t0 + step_time;
    u3 = u0 + step_time * f2;
    func_cal(f3, t3, u3);

    x0_fin = u0 +step_time * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0;
    time_fin = t3;
}




__global__ void dosth(double *data_result, int Nx, int Nt, double t0, double time_step_cal)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;

    double t0_cal = t0;

    if (ix < Nx)
    {
        for (int i0=0;i0<Nt;i0++)
        {
            if(i0!=0)
            {
                rk4_tryStep(t0_cal, data_result[ix+i0*Nx], t0_cal, data_result[ix+(i0-1)*Nx], time_step_cal);
            }
        }
    }

}



int main(int argc, char **argv)
{

    clock_t time0, time1;



    printf("nums of paras: %d\n",argc);
    for (int i0=1;i0<argc;i0++)
    {
        printf("Para[%d] = %s\n",i0,argv[i0]);
    }

    int nx = strtol(argv[1],NULL, 10);
    printf("the first number = %d\n",nx);

    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("using device: %s \n",deviceProp.name);

    int nElem_x = 1024;
    int nElem_t = 1024;
    printf("Vector size (%d,%d)\n", nElem_x,nElem_t);

    size_t nBytes = nElem_x * nElem_t * sizeof(double);


    // create 2d arrays on host
    double *h_result;
    h_result = (double *)malloc(nBytes);


    // create 2d arrays on GPU
    double *dev_result;
    cudaMalloc((void **) &dev_result, nBytes);

    // initialize data for the first row
    for (int i0=0;i0<nElem_x;i0++)
    {
        h_result[i0] = 0.1 + exp(-(double)(i0-nElem_x/2)*(double)(i0-nElem_x/2)/300/300);
    }


    // set GPU grid and block
    int dimx = nx;
    dim3 block;
    dim3 grid;


    block.x = dimx;
    block.y = 1;
    grid.x = (nElem_x + block.x - 1) / block.x;
    grid.y = 1;

    // copy host data to GPU
    cudaMemcpy(dev_result, h_result, nBytes, cudaMemcpyHostToDevice);

    time0 = clock();
    dosth<<<grid,block>>>(dev_result, nElem_x, nElem_t, 0, 1e-5*2);
    cudaThreadSynchronize();
    time1 = clock();

    cudaMemcpy(h_result, dev_result, nBytes, cudaMemcpyDeviceToHost);

    std::string ss = boost::lexical_cast<std::string>(123);
    cout << "string: "<< ss << "\n";
    double dd = boost::lexical_cast<double>(ss);
    cout << "number: "<<dd <<"\n";

    const string ss_file = ("result."+ boost::lexical_cast<std::string>(boost::format("%06d") % dd) + ".h5");

    //output the result to hdf5 file
////    {
////    const H5std_string FILE_NAME(ss_file);
////    const H5std_string DATASET_NAME("FloatArray");
////    H5File file( FILE_NAME, H5F_ACC_TRUNC);
////
////    hsize_t dimsf[2];
////    dimsf[0] = nElem_x;
////    dimsf[1] = nElem_t;
////    DataSpace dataspace(2, dimsf);
////
////    IntType datatype(PredType::NATIVE_DOUBLE);
////    datatype.setOrder(H5T_ORDER_LE);
////
////    DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
////    dataset.write(h_result, PredType::NATIVE_DOUBLE);
////    }

    output(h_result, "header_", ss_file, nElem_x, nElem_t);
    
    


    cudaFree(dev_result);

    free(h_result);

//    std::cout << "all done" <<std::endl;

//    std::cout << "results are: " <<std::endl <<std::endl;

//    for (int i0=0;i0<nElem_x;i0++)
//    {
//        std::cout << h_result[i0] << std::endl;
//    }


    std::cout<< "time: " << (float)(time1-time0)/CLOCKS_PER_SEC*1e3 << " ms"<< std::endl << std::endl;


    return 0;

}
