#include <cuda_runtime.h>


#include <cufft.h>
#include <cuComplex.h>

#include <time.h>

#define CUDART_M_PI 3.141592654



__global__ void calexp_nonlinear (cufftDoubleComplex *dev_exp_nonlinear, double k0, cufftDoubleComplex *dev_dn_nonlinear, double dz, int Nx, int Ny)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx= iy * Nx + ix;

    double exp_term_inner;
    double exp_term;

    if(ix < Nx && iy < Ny)
    {
        exp_term_inner = k0*dev_dn_nonlinear[idx].y*dz;
        exp_term = exp(exp_term_inner>1e-20?-exp_term:0);

        dev_exp_nonlinear[idx].x = exp_term * cos(k0*dev_dn_nonlinear[idx].x*dz);
        dev_exp_nonlinear[idx].y = exp_term * sin(k0*dev_dn_nonlinear[idx].x*dz);

//        dev_exp_nonlinear[idx].x = 1 * cos(k0*dev_dn_nonlinear[idx].x*dz);
//        dev_exp_nonlinear[idx].y = 1 * sin(k0*dev_dn_nonlinear[idx].x*dz);

    }


}

__global__ void calexp_linear (cufftDoubleComplex *dev_exp_linear, double k0, double nu_g, double dt, double dx,  double dz, int Nx, int Ny)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx= iy * Nx + ix;

    double freq_y;
    double freq_x;

    double omega_x;
    double omega_y;    

    if(ix < Nx && iy < Ny)
    {
        
        if(iy < Ny/2)
        {
            freq_y = 1.0/dt/2.0 / (double)(Ny/2) * iy;
        }
        else
        {
            freq_y = 1.0/dt/2.0 / (double)(Ny/2) * (Ny-iy);
        }
        if(ix < Nx/2)
        {
            freq_x = 1.0/dx/2.0 / (double)(Nx/2) * ix;
        }
        else
        {
            freq_x = 1.0/dx/2.0 / (double)(Nx/2) * (Nx-ix);
        }

        omega_y = 2.0 * CUDART_M_PI * freq_y;
        omega_x = 2.0 * CUDART_M_PI * freq_x;

        dev_exp_linear[idx].x = cos((nu_g/2.0*omega_y*omega_y - 1.0/2.0/k0*(omega_x*omega_x))*dz);
        dev_exp_linear[idx].y = sin((nu_g/2.0*omega_y*omega_y - 1.0/2.0/k0*(omega_x*omega_x))*dz);

    }


}


__global__ void initializeE0(cufftDoubleComplex *dev_phi_old, double E0max, double tau_pulse, double time_center, double w00, double dx, double dt, int Nx, int Ny)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx= iy * Nx + ix;

    double exp_time;

    if(ix < Nx && iy < Ny)
    {
        exp_time = exp(-(4.0*logf(2)/tau_pulse/tau_pulse*(iy*dt-time_center)*(iy*dt-time_center) ));

        dev_phi_old[idx].x = E0max * exp( -(ix-Nx/2)*(ix-Nx/2)*dx*dx/w00/w00 ) * exp_time;
        dev_phi_old[idx].y = 0;
    }


}

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

__global__ void calphi_omega_linear (cufftDoubleComplex *dev_phi_old_fft, cufftDoubleComplex *dev_exp_linear, int Nx, int Ny)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx= iy * Nx + ix;

    double result_real;
    double result_imag;


    if(ix < Nx && iy < Ny)
    {
        result_real = dev_phi_old_fft[idx].x * dev_exp_linear[idx].x - dev_phi_old_fft[idx].y * dev_exp_linear[idx].y;        
        result_imag = dev_phi_old_fft[idx].x * dev_exp_linear[idx].y + dev_phi_old_fft[idx].y * dev_exp_linear[idx].x;

        // it is normalized to 1/Nx/Ny, so that after ifft, the data is accurate
        dev_phi_old_fft[idx].x = result_real / (double)Nx / (double) Ny;
        dev_phi_old_fft[idx].y = result_imag / (double)Nx / (double) Ny;

    }


}


__global__ void calphi_time_nonlinear (cufftDoubleComplex *dev_phi_old, cufftDoubleComplex *dev_phi_old_fft, cufftDoubleComplex *dev_exp_nonlinear, int Nx, int Ny)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx= iy * Nx + ix;

    double result_real;
    double result_imag;


    if(ix < Nx && iy < Ny)
    {
        result_real = dev_phi_old_fft[idx].x * dev_exp_nonlinear[idx].x - dev_phi_old_fft[idx].y * dev_exp_nonlinear[idx].y;        
        result_imag = dev_phi_old_fft[idx].x * dev_exp_nonlinear[idx].y + dev_phi_old_fft[idx].y * dev_exp_nonlinear[idx].x;

        dev_phi_old[idx].x = result_real;
        dev_phi_old[idx].y = result_imag;

    }


}


__global__ void calphi_abs2 (double *dev_phi_abs2, cufftDoubleComplex *dev_phi_old, int Nx, int Ny)
{

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx= iy * Nx + ix;

    double phi_abs_cal;

    if(ix < Nx && iy < Ny)
    {
        phi_abs_cal = cuCabs(dev_phi_old[idx]);

        dev_phi_abs2[idx] = phi_abs_cal * phi_abs_cal;

    }


}
