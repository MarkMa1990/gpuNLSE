//#include <stdio.h>
//#define _USE_MATH_DEFINES
//#include <math.h>
#include <cuda_runtime.h>

//#include <stdlib.h>

//#include <string>

#include <cufft.h>
#include <cuComplex.h>

#define CUDART_M_PI 3.141592654




__device__ void get_MPIbyIntensity (double &mpi_get, double inten_cal, double *mpi_inten_cal, double *mpi_ionization_cal)
{

    if(inten_cal > 1.0e19)
    {
//        std::cout <<"MPI, Intensity should not be higher than 1e19" << std::endl;
        mpi_get = -1;
    }

    else
    {

    int i=0;
    while(inten_cal > mpi_inten_cal[i+1]) i++;
    mpi_get = (inten_cal-mpi_inten_cal[i])/(mpi_inten_cal[i+1]-mpi_inten_cal[i])*(mpi_ionization_cal[i+1]-mpi_ionization_cal[i]) + mpi_ionization_cal[i];
//    mpi_get = 0;
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


__device__ void func_cal(double &rho_e_result, double rho_e_0, double inten_cal, double n_real_cal, double *mpi_inten_cal, double *mpi_ionization_cal, double *paras)
{
    double wpi_cal0;
    get_MPIbyIntensity(wpi_cal0, inten_cal, mpi_inten_cal, mpi_ionization_cal);
    // e^2* tau_colli /( me_eff(omega_wave^2*tau_colli^2+1.0)* c0 *eps_0)
    double sigma_ab_0 = paras[7]*paras[7]*paras[10]/(paras[5]*(paras[14]*paras[14]*paras[10]*paras[10]+1.0)*paras[6]*paras[3]);
    // (wpi + sigma_ab_0/n0/Ueff * I * rho_e)*(1.0-rho_e/rho_max) - rho_e/tau_trap_e
    rho_e_result = (wpi_cal0 + sigma_ab_0/paras[0]*(1.0 + paras[5]/paras[4])/(paras[2]+paras[7]*paras[7]/4.0/paras[5]/paras[14]/paras[14]*inten_cal/(n_real_cal*paras[6]*paras[3]/2.0))*inten_cal*rho_e_0) * (1.0 - rho_e_0/paras[9]) - rho_e_0 / paras[11];
}

__device__ void rk4_tryStep(double &rho_e_x_fin, double rho_e_x_init, double step_time, double inten_cal, double n_real_cal, double *mpi_inten_cal, double *mpi_ionization_cal, double *paras)
{
    double u0, u1, u2, u3;
    double f0, f1, f2, f3;

    u0 = rho_e_x_init;
    func_cal(f0, u0, inten_cal, n_real_cal, mpi_inten_cal, mpi_ionization_cal, paras);

    u1 = u0 + step_time * f0 / 2.0;
    func_cal(f1, u1, inten_cal, n_real_cal, mpi_inten_cal, mpi_ionization_cal, paras);

    u2 = u0 + step_time * f1 / 2.0;
    func_cal(f2, u2, inten_cal, n_real_cal, mpi_inten_cal, mpi_ionization_cal, paras);

    u3 = u0 + step_time * f2;
    func_cal(f3, u3, inten_cal, n_real_cal, mpi_inten_cal, mpi_ionization_cal, paras);

    rho_e_x_fin = u0 + step_time * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0;
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


__global__ void calRho_X(double *data_rho_e_xt, double *data_n_real, double *data_n_imag, cufftDoubleComplex *data_dn_nonlinear, int Nx, int Nt, double time_step_cal, cufftDoubleComplex *phi_old, double *mpi_inten_cal, double *mpi_ionization_cal, double *paras)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;

    double inten_now = 0;
    double n_real_now = 0;
    cufftDoubleComplex eps_time_cal;
    double rho_old;
    double rho_new;
    // define jj = complex(0,1);
//    cufftDoubleComplex jj;
//    jj.x = 0;
//    jj.y = 1.0;

    double sigma_ab_0 = paras[7]*paras[7]*paras[10]/(paras[5]*(paras[14]*paras[14]*paras[10]*paras[10]+1.0)*paras[6]*paras[3]);



    double wpi_cal0;


    if (ix < Nx)
    {
        for (int i0=0;i0<Nt;i0++)
        {
            if(i0==0)
            {
                data_n_real[ix] = paras[0];
                data_n_imag[ix] = 0;
                data_rho_e_xt[ix] = 0;

                data_dn_nonlinear[ix].x = 0;
                data_dn_nonlinear[ix].y = 0;

                
            }
            else
            {
                //calculate eps_time_cal
                n_real_now = data_n_real[ix+(i0-1)*Nx];
                inten_now = n_real_now * paras[6] * paras[3] / 2.0 * cuCabs(phi_old[ix+i0*Nx]) * cuCabs(phi_old[ix+i0*Nx]);               

                rho_old = data_rho_e_xt[ix+(i0-1)*Nx];

                // do rk4 one step
                rk4_tryStep(rho_new, rho_old, time_step_cal, inten_now, n_real_now, mpi_inten_cal, mpi_ionization_cal, paras);
                data_rho_e_xt[ix+i0*Nx] = rho_new;
                
                // calculate n_real, and n_imag
                //eps = n0*n0 + 2.0*n0*n2*intensity_time_cal - c0*sigma_ab_0/omega_wave*(omega_wave*tau_colli_e - jj) * rho_electron_cal[ii];
                eps_time_cal.x = paras[0]*paras[0] + 2.0*paras[0]*paras[1]*inten_now - paras[6]*sigma_ab_0/paras[14]*(paras[14]*paras[10])*rho_new;
                eps_time_cal.y = paras[6]*sigma_ab_0/paras[14]*(1)*rho_new;
                data_n_real[ix+i0*Nx] =(double)(sqrtf((float)cuCabs(eps_time_cal) + (float)eps_time_cal.x/sqrtf(2)));
                data_n_imag[ix+i0*Nx] =(double)(sqrtf((float)cuCabs(eps_time_cal) - (float)eps_time_cal.x/sqrtf(2)));

                // calculate nonlinear term
////      delta_n_cal[ii][0] = n2*n_real_cal[ii]*eps_0*c0/2.0*(Gau_inten_cal[ii][0]*Gau_inten_cal[ii][0]+Gau_inten_cal[ii][1]*Gau_inten_cal[ii][1])
////          - sigma_ab_0/n0*omega_wave*tau_colli_e/2.0/k0*rho_electron_cal[ii];


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

                data_dn_nonlinear[ix+i0*Nx].x = paras[1]*data_n_real[ix+i0*Nx]*paras[3]*paras[6]/2.0*cuCabs(phi_old[ix+i0*Nx])*cuCabs(phi_old[ix+i0*Nx])
                    - sigma_ab_0/paras[0]*paras[14]*paras[10]/2.0/(2.0*CUDART_M_PI/paras[13])*rho_new;

////      delta_n_cal[ii][1] = wpi_cal*Ug/n_real_cal[ii]/eps_0/c0/k0/(Gau_inten_cal[ii][0]*Gau_inten_cal[ii][0]+Gau_inten_cal[ii][1]*Gau_inten_cal[ii][1])
////          + sigma_ab_0/n0/2.0/k0*rho_electron_cal[ii];

                get_MPIbyIntensity(wpi_cal0, inten_now, mpi_inten_cal, mpi_ionization_cal);
                data_dn_nonlinear[ix+i0*Nx].y = wpi_cal0*paras[2]/data_n_real[ix+i0*Nx]/paras[3]/paras[6]/(2.0*CUDART_M_PI/paras[13])/cuCabs(phi_old[ix+i0*Nx])/cuCabs(phi_old[ix+i0*Nx])
                    + sigma_ab_0/paras[0]/2.0/(2.0*CUDART_M_PI/paras[13])*rho_new;


            }
        }
    }

}



