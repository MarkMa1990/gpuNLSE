# gpuNLSE

## Introduction
<<<2D+1 NLSE with GPU acceleration>>>

This project aims at solving the nonlinear schrodinger equation (NLSE) coupled with rate equation, which can be applied for simulting ultrafast lasers interation with materials such as glass, porous glass, silica and etc.

The related codes are developed during my Ph.D. studying, if you are interested in this project and would like to contribute, please send me a message. Any suggestions, improvements, or comments are welcomed.

The name 2D relates to x-y coordinates. Light is incident in +z axis. 

## Code
```
Language: The code was developed in CUDA C/C++. 
Compile: Windows, Ubuntu with CMake
Dependencies: HDF5 (for saving result), Boost (just for string handling)
```
## Reference

if you are using the code at current or modified version derived from this project, and if you want to cite the related works, please find as follows:

```
@article{ma2021numerical,
  title={Numerical study of laser micro-and nano-processing of nanocomposite porous materials},
  author={Ma, Hongfeng},
  journal={arXiv preprint arXiv:2103.07334},
  year={2021}
}

@article{ma2017well,
  title={Well-controlled femtosecond laser inscription of periodic void structures in porous glass for photonic applications},
  author={Ma, Hongfeng and Zakoldaev, Roman A and Rudenko, Anton and Sergeev, Maksim M and Veiko, Vadim P and Itina, Tatiana E},
  journal={Optics express},
  volume={25},
  number={26},
  pages={33261--33270},
  year={2017},
  publisher={Optical Society of America}
}
```
