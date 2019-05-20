cmake_minimum_required(VERSION 3.8)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0)
find_package(CUDA 9.0 REQUIRED)
#cuda_add_executable(routines routines.cu)
cuda_add_executable(main_loop main_loop.cu)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcufft)
cuda_add_cufft_to_target(main_loop)

# HDF5
#set(HDF5_ROOT_DIR /home_expes/tools/libraries/hdf5)
find_package(HDF5 COMPONENTS CXX HL)
# include_directories("/home_expes/tools/libraries/hdf5/include")
include_directories(${HDF5_INCLUDE_DIRS})

# BOOST
#set(BOOST_ROOT /home_expes/tools/boost/boost_1_66_0)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

message("${CUDA_LIBRARIES}")

# for linking
target_link_libraries(main_loop ${CUDA_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(main_loop -lcudart -lcufft ${HDF5_LIBRARIES} ${Boost_LIBRARIES})

