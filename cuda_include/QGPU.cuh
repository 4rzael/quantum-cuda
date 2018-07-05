/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QGPU.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#pragma once

# include <valarray>
# include <complex>

/**
 * Included to use QCUDA::CUDAGPU class as an attribute in QGPU::GPU class.
 * Therefore, this attribute will allow us to communicate with the GPU
 * through his methods.
 */
# include "QCUDA.cuh"

/**
* @brief Namespace QGPU contains a template class version of GPUExecutor
* called GPU.
*
* This Namespace is composed as the same as the class 'GPUExecutor', except
* that this one is entirely developed to let the user to select the precision he
* wants to use.
* However, this precision is limited. According to Nvidia, CUDA can only either 
* handle the 'double', or 'float' number type.
* This Header is not actually used in this project, but the developers decided
* to let this file in the project, based on the evolution of the project.
*/
namespace QGPU {

  /**
   * @brief GPU is a template class composed of linear algebra operations that
   * will be performed on the GPU.
   *
   * This class will be embedded in the 'ExecutorManager' singleton, to wether
   * enable the 'CPUExecutor' or the 'GPUExecutor' according to the user's
   * computer.
   */
  template<typename T>
  class GPU {
  private:
    /**
     * Attribute that will allow those operations below to communicate with
     * the GPU.
     */
    QCUDA::CUDAGPU<T>	cgpu_;
  public:
    /**
     * Default contructor of GPU class.
     */
    GPU();
    /**
     * Destructor of GPU class
     */
    virtual ~GPU();
  public:
    /**
     * @brief Performs an addition between two
     * std::valarray<std::complex<T>>, i.e. Matrices.
     *
     * @param a A matrix content.
     * @param b B matrix content.
     * @return The addition between matrices a and b as a pointer of
     * std::valarray<std::complex<T>>.
     *
     * Those Matrices will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
     */
    std::valarray<std::complex<T>>*
    add(const std::valarray<std::complex<T>>* a,
	const std::valarray<std::complex<T>>* b);

    /**
     * @brief Performs a dot product between
     * std::valarray<std::complex<T>>, i.e. Matrices.
     *
     * @param a A matrix content.
     * @param b B matrix content.
     * @param ma A matrix m dimension.
     * @param mb B matrix m dimension.
     * @param na A matrix n dimension.
     * @param mb B matrix n dimension.
     * @return The dot product result as a pointer
     * std::valarray<std::complex<T>>.
     *
     * Those Matrices will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
     */
    std::valarray<std::complex<T>>*
    dot(const std::valarray<std::complex<T>>* a,
	const std::valarray<std::complex<T>>* b,
	const int ma,
	const int mb,
	const int na,
	const int nb);

    /**
     * @brief Performs a kroenecker product between two
     * std::valarray<std::complex<T>>, i.e. Matrices..
     *
     * @param a A matrix content.
     * @param b B matrix content.
     * @param ma A matrix m dimension.
     * @param mb B matrix m dimension.
     * @return The dot product result as a pointer of
     * std::valarray<std::complex<T>>.
     *
     * Those Matrices will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
     */
    std::valarray<std::complex<T>>*
    kron(const std::valarray<std::complex<T>>* a,
	 const std::valarray<std::complex<T>>* b,
	 const int ma,
	 const int mb);

    /**
     * @brief Compute the trace of a std::valarray<std::complex<T>>,
     * i.e. Matrix.
     *
     * @param a A matrix content.
     * @param m A matrix m dimension.
     * @return The trace as a number complex of std::complex<T>.
     *
     * The Matrix will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
     */
    std::complex<T>
    trace(const std::valarray<std::complex<T>>*,
       const int);

    /**
     * @brief Compute the transpose of a std::valarray<std::complex<T>>,
     * i.e. Matrix..
     *
     * @param a A matrix content.
     * @param m A matrix m dimension.
     * @param n A matrix n dimension.
     * @return The transpose as a pointer of std::valarray<std::complex<T>>.
     *
     * The Matrix will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
     */
    std::valarray<std::complex<T>>*
    transpose(const std::valarray<std::complex<T>>* a,
	      const int m,
	      const int n);

    /**
     * @brief Compute the normalized std::valarray<std::complex<T>>,
     * i.e. Matrix..
     *
     * @Param a A matrix content.
     * @return The normalized matrix as a pointer of
     * std::valarray<std::complex<T>>
     *
     * The Matrix will be converted in order to fit with the requirements
     * to run on an Nvidia's GPU.
     */
    std::valarray<std::complex<T>>*
    normalize(std::valarray<std::complex<T>>* a);
  };


  /**
   * Definition of GPU's constructor.
   *
   * The constructor contains nothing for the moment in its body, except an
   * initializer list that will just call initilize our 'cgpu_' attribute
   * by calling his constructor.
   */
  template<typename T>
  GPU<T>::GPU()
    : cgpu_()
  {};


  /**
   * Definition of GPU's destructor.
   *
   * The destructor is actually defined with the c++ 'default' keyword.
   */
  template<typename T>
  GPU<T>::~GPU() = default;


  /**
   * Definition of GPU's 'add' method.
   *
   * The 'add' method is actually based on these steps:
   *
   * - The first part is about the 'initThrustHostVec':
   * -- The first part is about to convert the received matrices with a 
   * specific container, i.e thrust::(device/host)_vector container
   * thanks to the CUDA's THRUST library, and then copy the content of those
   * matrices to in the thrust::host_vector container.
   * This library is especially designed to allow the user to use specialized
   * container with CUDA, and therefore these containers some optimizations
   * operations between host and device parts.
   *
   * - The second part is about the 'assignHostToDevice':
   * -- The content of thrust::host_vector containers will then asssigned
   * to the thrust::device_vector containers.
   * The lines of code related to this are actually commented since we are using
   * a containers for the beginning of our project.
   *
   * - The third part is about the 'convertDeviceToCUDAType':
   * -- Once the thrust::(device/host)_vector initialized, will then convert
   * those containers again to a customized container that will be send in
   * the CUDA kernel.
   *
   * - The fourth part is about 'performAddOnGPU':
   * -- This method will performed all the requirements to execute an
   * addition, therefore, calling the addition kernel.
   *
   * WARNING: The Method contains outputs of debug, and are just temporary
   * until the method doesn't reach its final definition.
   */
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::add(const std::valarray<std::complex<T>>* m1,
	      const std::valarray<std::complex<T>>* m2) {
    std::valarray<std::complex<T>>* ptr;

    std::cout << "== OUTPUT 'add' FUNCTION HEAD ==" << std::endl << std::endl;
    this->cgpu_.initThrustHostVec((*m1),
    				  (*m2),
    				  QCUDA::DeviceVectors::DEVICE_VECTORS);
    // this->cgpu_.assignHostToDevice(QCUDA::DeviceVectors::DEVICE_VECTOR_A);
    // this->cgpu_.assignHostToDevice(QCUDA::DeviceVectors::DEVICE_VECTOR_B);
    this->cgpu_.convertDeviceToCUDAType(QCUDA::DeviceVectors::DEVICE_VECTOR_A);
    this->cgpu_.convertDeviceToCUDAType(QCUDA::DeviceVectors::DEVICE_VECTOR_B);
    ptr = this->cgpu_.performAddOnGPU();
    std::cout << std::endl << "== OUTPUT 'add' FUNCTION TAIL ==" << std::endl;
    return (ptr);
  };

  /**
   * Definition of GPU's 'add' method.
   * The behaviour is the same as add (steps 1, 2, 3) except for the fourth
   * part, where dot kernel is called.
   * 
   * WARNING: The Method contains outputs of debug, and are just temporary
   * until the method doesn't reach its final definition.
   */
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::dot(const std::valarray<std::complex<T>>*,
	      const std::valarray<std::complex<T>>*,
	      const int,
	      const int,
	      const int,
	      const int) {
    return (nullptr);
  };


  /**
   * Definition of GPU's 'kron' method.
   * The behaviour is the same as add (steps 1, 2, 3) except for the fourth
   * part, where kron kernel is called.
   * 
   * WARNING: The Method contains outputs of debug, and are just temporary
   * until the method doesn't reach its final definition.
   */  
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::kron(const std::valarray<std::complex<T>>*,
	       const std::valarray<std::complex<T>>*,
	       const int,
	       const int) {
    return (nullptr);
  };


  /**
   * Definition of GPU's 'trace' method.
   *
   * The behaviour is the same as add (steps 1, 2, 3) except for the fourth
   * part, where trace kernel is called.
   *
   * WARNING: The Method contains outputs of debug, and are just temporary
   * until the method doesn't reach its final definition.
   */
  template<typename T>
  std::complex<T>
  GPU<T>::trace(const std::valarray<std::complex<T>>*,
	     const int) {
    std::complex<T>	tmp;
    return (tmp);
  };


  /**
   * Definition of GPU's 'transpose' method.
   *
   * The behaviour is the same as add (steps 1, 2, 3) except for the fourth
   * part, where dot transpose is called.
   * 
   * WARNING: The Method contains outputs of debug, and are just temporary
   * until the method doesn't reach its final definition.
   */
  template<typename T>
  std::valarray<std::complex<T>>*
  GPU<T>::transpose(const std::valarray<std::complex<T>>* a,
		    const int m,
		    const int n) {

  };


  /**
   * Definition of GPU's 'normalize' method.
   *
   * The behaviour is the same as add (steps 1, 2, 3) except for the fourth
   * part, where normalize kernel is called.
   *
   * WARNING: The Method contains outputs of debug, and are just temporary
   * until the method doesn't reach its final definition.
   */
  template<typename T>
  std::valarray<std::complex<T>>*
  normalize(std::valarray<std::complex<T>>* a) {
    return (nullptr);
  }

};
