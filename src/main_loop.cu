#include <iostream>
#include <cmath>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>

#include <stdlib.h>

#include <string>
#include "H5Cpp.h"

#include <cufft.h>
#include <cuComplex.h>

#include <time.h>

#include "rk4Method.cu"
#include "calLNexp.cu"

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

//using namespace std;
using namespace H5;

void output_data_real(double *data_out, std::string header_name, int step_name, int nElem_x, int nElem_t)
{
    const std::string filename_str = (
                            header_name + 
                            "." 
                            +  boost::lexical_cast<std::string>(boost::format("%06d") % step_name)
                            + ".h5"
                         );

    const H5std_string FILE_NAME(filename_str);
    const H5std_string DATASET_NAME("FloatArray");
    H5File file( FILE_NAME, H5F_ACC_TRUNC);

    hsize_t dimsf[2];
    dimsf[0] = nElem_x;
    dimsf[1] = nElem_t;
    DataSpace dataspace(2, dimsf);

    IntType datatype(PredType::NATIVE_DOUBLE);
    datatype.setOrder(H5T_ORDER_LE);

    DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
    dataset.write(data_out, PredType::NATIVE_DOUBLE);
    
}


void host_readMPI(double *mpi_inten, double *mpi_ionization)
{

  FILE *fp1 = fopen("./glass_reduced2.txt","r");

  if (!fp1)
      std::cout << "FILE not exists! Please check it again!" << std::endl;

  else
  {
    int i_num=0;
    double data_read_inten, data_read_ioin;
    while(!feof(fp1))
    {
        fscanf(fp1,"\t%lf \t%lf \n",&data_read_inten, &data_read_ioin);
        mpi_inten[i_num] = data_read_inten;
        mpi_ionization[i_num] = data_read_ioin;
        i_num +=1;
    }

    std::cout << "reading MPI_keldysh success..." << std::endl;
  }

    fclose(fp1);
}


int main()
{
    // init device
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("using device: %s \n",deviceProp.name);


    // space discretization
    double dx = 200e-9;
    int Nx_cal = 1024;
    
    // time discretization
    double dt = 0.8e-15;
    int Nt_cal = 1024;

    // space in z
    double dz = 120e-9*5;
    int Nz_cal = 1024*2;


    // init laser paras on device
    double *host_laser_paras;
    double *dev_laser_paras;

    host_laser_paras    = new double[15];
    cudaMalloc((void **) &dev_laser_paras,        15*sizeof(double));

   
    // do something
        // paras
        //  0:  n0
        //  1:  n2
        //  2:  Ug
        //  3:  eps_0
        //  4:  me
        //  5:  me_eff
        //  6:  c0, light speed in vacuum
        //  7:  e_charge, elementary charge
        //  8:  k_2
        //  9:  rho_max
        //  10:  tau_colli_e
        //  11:  tau_trap_e
        //  12:  belta_avalanche

        //  13:  lmv_wave_vacuum
        //  14:  omega_wave
    host_laser_paras[0] = 1.342;
    host_laser_paras[1] = 3.54e-20;
    host_laser_paras[2] = 9.0*1.6021766e-19;
    host_laser_paras[3] = 8.854e-12;
    host_laser_paras[4] = 0.91e-30;
    host_laser_paras[5] = 0.86*host_laser_paras[4];
    host_laser_paras[6] = 3.0e8;
    host_laser_paras[7] = 1.6021766e-19;
    host_laser_paras[8] = 361e-28;
    host_laser_paras[9] = 2.2e28;
    host_laser_paras[10] = 1.0e-15;
    host_laser_paras[11] = 150e-15;
    host_laser_paras[12] = 3.8e-4;
    host_laser_paras[13] = 515e-9;
    host_laser_paras[14] = 2.0*M_PI*host_laser_paras[6]/host_laser_paras[13];

    cudaMemcpy(dev_laser_paras, host_laser_paras, 15*sizeof(double), cudaMemcpyHostToDevice);

    // allocate mem

        // data on host
    double *host_rho_e;
    double *host_n_real;
    double *host_n_imag;
    double *host_phi_abs2;

    cufftDoubleComplex *host_phi_old;

        // data on GPU
    double *dev_rho_e;
    double *dev_n_real;
    double *dev_n_imag;
    double *dev_phi_abs2;

    cufftDoubleComplex *dev_phi_old;
    cufftDoubleComplex *dev_phi_old_fft;
    
            // allocate mem host
    host_rho_e      = new double[Nx_cal*Nt_cal];
    host_n_real     = new double[Nx_cal*Nt_cal];
    host_n_imag     = new double[Nx_cal*Nt_cal];
    host_phi_abs2   = new double[Nx_cal*Nt_cal];

    host_phi_old   = new cufftDoubleComplex[Nx_cal*Nt_cal];

            // allocate mem on GPU
    cudaMalloc((void **) &dev_rho_e,        Nx_cal*Nt_cal*sizeof(double));
    cudaMalloc((void **) &dev_n_real,        Nx_cal*Nt_cal*sizeof(double));
    cudaMalloc((void **) &dev_n_imag,        Nx_cal*Nt_cal*sizeof(double));
    cudaMalloc((void **) &dev_phi_abs2,        Nx_cal*Nt_cal*sizeof(double));

    cudaMalloc((void **) &dev_phi_old,        Nx_cal*Nt_cal*sizeof(cufftDoubleComplex));
    cudaMalloc((void **) &dev_phi_old_fft,        Nx_cal*Nt_cal*sizeof(cufftDoubleComplex));

    // nonlinear part
    cufftDoubleComplex *dev_dn_nonlinear;
    cudaMalloc((void **) &dev_dn_nonlinear,        Nx_cal*Nt_cal*sizeof(cufftDoubleComplex));

    // exp_nonlinear
    cufftDoubleComplex *dev_exp_nonlinear;
    cudaMalloc((void **) &dev_exp_nonlinear,        Nx_cal*Nt_cal*sizeof(cufftDoubleComplex));

    // exp_linear
    cufftDoubleComplex *dev_exp_linear;
    cudaMalloc((void **) &dev_exp_linear,        Nx_cal*Nt_cal*sizeof(cufftDoubleComplex));


    // storage of MPI
    double *host_mpi_inten;
    double *host_mpi_ionization;
    //      device 
    double *dev_mpi_inten;
    double *dev_mpi_ionization;

    host_mpi_inten      = new double[301];
    host_mpi_ionization = new double[301];

    cudaMalloc((void **) &dev_mpi_inten,        301*sizeof(double));
    cudaMalloc((void **) &dev_mpi_ionization,   301*sizeof(double));

    // read MPI
    host_readMPI(host_mpi_inten, host_mpi_ionization);
    // copy host data to GPU
    cudaMemcpy(dev_mpi_inten, host_mpi_inten, 301*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mpi_ionization, host_mpi_ionization, 301*sizeof(double), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();


    // initialize E0
    double w00 = 35e-6/2.0; //waise of beam
    double E_pulse = 0.3e-6; //laser pulse energy
    double tau_pulse = 300e-15;
    double P0_max = E_pulse / tau_pulse;
    double I0max = P0_max / (M_PI*w00*w00/2.0);
    double E0max = std::sqrt(2.0*I0max/(host_laser_paras[0]*3e8*host_laser_paras[3]));

    // k00 = 2pi/lambda_vacuum
    double k00 = 2.0*M_PI/host_laser_paras[13];

    // define time part of laser
    double t_center = dt*(double)Nt_cal/3.0;
   
        // define GPU paras for 2D
    int dimx_2d = 32;
    int dimy_2d = 32;

    dim3 block_2d;
    dim3 grid_2d;

    block_2d.x = dimx_2d;
    block_2d.y = dimy_2d;
    grid_2d.x = (Nx_cal + block_2d.x - 1) / block_2d.x;
    grid_2d.y = (Nt_cal + block_2d.y - 1) / block_2d.y;

        // define GPU paras for 1D, RK4 method
    int dimx_1d = 32;

    dim3 block_1d;
    dim3 grid_1d;

    block_1d.x = dimx_1d;
    block_1d.y = 1;
    grid_1d.x = (Nx_cal + block_1d.x - 1) / block_1d.x;
    grid_1d.y = 1;

    // start
    initializeE0<<<grid_2d, block_2d>>>(dev_phi_old, E0max, tau_pulse, t_center, w00, dx, dt, Nx_cal, Nt_cal);
    cudaThreadSynchronize();

    // generate linear exp
    calexp_linear<<<grid_2d, block_2d>>>(dev_exp_linear, k00, host_laser_paras[8], dt, dx, dz, Nx_cal, Nt_cal);
    cudaThreadSynchronize();

    clock_t time0, time1;
    clock_t time0_all, time1_all;
  
    // output once
    // calculate phi_abs2
    calphi_abs2<<<grid_2d,block_2d>>>(dev_phi_abs2, dev_phi_old, Nx_cal, Nt_cal);
    cudaThreadSynchronize();

    cudaMemcpy(host_phi_abs2,  dev_phi_abs2,  Nx_cal*Nt_cal*sizeof(double), cudaMemcpyDeviceToHost);
    output_data_real(host_phi_abs2, "phi_abs2", 0, Nx_cal, Nt_cal);

    // create CUDA FFT plan
    cufftHandle dev_plan_fft2d;
    cufftPlan2d(&dev_plan_fft2d, Nx_cal, Nt_cal, CUFFT_Z2Z);

    for (int iz00=1;iz00<Nz_cal;iz00++)
    {
        std::cout << std::endl
                  <<" \t ------ I am at step: ------"<<iz00 <<std::endl;
        
        time0_all = clock();

        // do FFT, for linear part update
        time0 = clock();
        cufftExecZ2Z(dev_plan_fft2d, dev_phi_old, dev_phi_old_fft, CUFFT_FORWARD);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : FFT_FORWARD \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);

        time0 = clock();
        calphi_omega_linear<<<grid_2d, block_2d>>>(dev_phi_old_fft, dev_exp_linear, Nx_cal, Nt_cal);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : FFT multipy exp(linear part) \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);
        
        time0 = clock();
        cufftExecZ2Z(dev_plan_fft2d, dev_phi_old_fft, dev_phi_old_fft, CUFFT_INVERSE);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : FFT_INVERSE \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);

        // nonlinear part update
        time0 = clock();
        calRho_X<<<grid_1d, block_1d>>>(dev_rho_e, dev_n_real, dev_n_imag, dev_dn_nonlinear, Nx_cal, Nt_cal, dt, dev_phi_old, dev_mpi_inten, dev_mpi_ionization, dev_laser_paras);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : calculate delta_n \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);

        time0 = clock();
        calexp_nonlinear<<<grid_2d, block_2d>>>(dev_exp_nonlinear, k00, dev_dn_nonlinear, dz, Nx_cal, Nt_cal);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : calculate exp(nonlinear) \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);

        time0 = clock();
        calphi_time_nonlinear<<<grid_2d, block_2d>>>(dev_phi_old, dev_phi_old_fft, dev_exp_nonlinear, Nx_cal, Nt_cal);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : calculate IFFT(phi*exp(D))*exp(N) \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);

        // calculate phi_abs2
        time0 = clock();
        calphi_abs2<<<grid_2d,block_2d>>>(dev_phi_abs2, dev_phi_old, Nx_cal, Nt_cal);
        cudaThreadSynchronize();
        time1 = clock();
        printf(" %0.1f ms : get |phi|^2 \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);
        
        if (iz00 % 20 ==0)
        {
        //
        time0 = clock();
        cudaMemcpy(host_n_real, dev_n_real, Nx_cal*Nt_cal*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_n_imag, dev_n_imag, Nx_cal*Nt_cal*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_rho_e,  dev_rho_e,  Nx_cal*Nt_cal*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_phi_abs2,  dev_phi_abs2,  Nx_cal*Nt_cal*sizeof(double), cudaMemcpyDeviceToHost);
        time1 = clock();
        printf(" %0.1f ms : copy n_real, n_imag, rho_e, phi_abs2 to host \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);

        // output
        time0 = clock();
        output_data_real(host_n_real,   "n_real",   iz00, Nx_cal, Nt_cal);
        output_data_real(host_n_imag,   "n_imag",   iz00, Nx_cal, Nt_cal);
        output_data_real(host_rho_e,    "rho_e",    iz00, Nx_cal, Nt_cal);
        output_data_real(host_phi_abs2, "phi_abs2", iz00, Nx_cal, Nt_cal);
        time1 = clock();
        printf(" %0.1f ms : output to hdf5 \n", (float)(time1-time0)/CLOCKS_PER_SEC*1e3);
        }


        time1_all = clock();
        printf(" ---------------------- \n");
        printf(" OVERALL COST: %0.1f ms \n\n", (float)(time1_all-time0_all)/CLOCKS_PER_SEC*1e3);

        

    }

         printf(" Everything is done. \n");   

   // release GPU mem
   cudaFree( dev_mpi_ionization );
   cudaFree( dev_mpi_inten );

   cudaFree( dev_exp_linear );
   cudaFree( dev_exp_nonlinear );
   cudaFree( dev_dn_nonlinear );

   cudaFree( dev_phi_old );
   cudaFree( dev_phi_old_fft );
   cudaFree( dev_phi_abs2 );

   cudaFree( dev_rho_e );
   cudaFree( dev_n_real );
   cudaFree( dev_n_imag );

   cudaFree( dev_laser_paras );

   // release host memmory

   delete [] host_mpi_ionization;
   delete [] host_mpi_inten;

   delete [] host_phi_old;
   delete [] host_phi_abs2;

   delete [] host_rho_e;
   delete [] host_n_real;
   delete [] host_n_imag;

   delete [] host_laser_paras;

}
