/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: GPUProperties.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

#pragma once

# include <string>
# include <cstring>
# include "QCUDA_utils.cuh"

//! \file GPUProperties.cuh
//! \brief GPUProperties.cuh contains the declaration of the
//!        class GPUProperties, and all other elements related to this class.
//!


namespace QCUDA {
  //! \class GPUProperties
  //! \brief GPUProperties fetchs all the Nvidia GPUs on the system, and
  //!        encapsulates them.
  //!
  //! Once the Nvidia GPUs encapsulated, the user -based on GPUCriteria enum-
  //! can now select which GPU the computation will be performed.
  //! \see GPUCriteria
  //!
  class GPUProperties {
  private:
    //! \private
    //! \brief Contains the number of Nvidia GPUs found on the system.
    //!
    int			nbGPUs_;


    //! \private
    //! \brief Contains the ID of the chosen Nvidia GPU, where all the
    //!        computation will be performed.
    //!
    //! 
    int			actualDeviceID_;

  protected:
    //! \protected
    //! \brief Contains all properties of the chosen Nvidia GPU, where all the
    //!        computation will be performed.
    //!
    //! For more information
    //! <a href="CudaDeviceProp">https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html</a>
    //! The attribute is set as protected, in order to let the class
    //! GPUDimemsions retrieves all the information to correctly set
    //! the size of the grid and the numnber of threads per blocks according to
    //! the Nvidia GPU's characteristics.
    //!
    cudaDeviceProp	actualDeviceProp_;


  private:
    //! \private
    //! \brief error_ is a wrapper, with specific attributes and methods
    //!        that contains the error encountered with the CUDA API.
    //!
    errorHandler_t	error_;
  public:
    //! \public
    //! \brief Constructor of the class GPUProperties.
    //!
    //! \param c Correponds to which criteria the instance of the GPU part will
    //!          be initiazed.
    __host__
    GPUProperties(const QCUDA::GPUCriteria c);


    //! \public
    //! \brief Destructor of the class GPUPropeties.
    //!
    __host__
    ~GPUProperties();
  public:
    //! \public
    //! \brief This method will iterate on each fetched Nvidia GPU in order
    //!        to check the eligibility of the Nvidia GPU with the criteria.
    //!
    //! \param c Corresponds to the chosen criteria.
    __host__
    void	selectGPUFromCriteria(const QCUDA::GPUCriteria c);


    //! \public
    //! \brief print all the key properties of the chosen Nvidia GPU.
    //!
    __host__
    void	dumpGPUProperties() const;

    //! \public
    //! \brief
    //!
    __host__
    const cudaDeviceProp&	getDeviceProp() const;
  private:
    //! \private
    //! \brief Acts as a "gateway" -switch statement- thanks to c param. I.e.
    //!        the method will check which kind of criteria is, and will send
    //!        it to the right method.
    //!
    //! \param c Corresponds to the chosen criteria. 
    //! \param prop Corresponds to the properties of the chosen Nvidia GPU based
    //!        on criteria c.
    //! \param id Corresponds to the id of the chosen Nvidia GPU.
    //!
    __host__
    int	analysePropForCriteria(const QCUDA::GPUCriteria c,
			       cudaDeviceProp* prop,
			       int id);


    //! \private
    //! \brief This method will iterate through Nvidia GPU and will select the
    //!        first Nvidia GPU double type compliant.
    //! \param prop Corresponds to the properties of the chosen Nvidia GPU based
    //!        on criteria c.
    //! \param id Corresponds to the id of the chosen Nvidia GPU.
    //! 
    __host__
    int	doubleTypeCompliant(cudaDeviceProp* prop, int id);

    
    //! \private
    //! \brief This method will iterate through Nvidia GPU and select the Nvidia
    //!        with the highest Compute Capability.
    //! \param prop Corresponds to the properties of the chosen Nvidia GPU based
    //!        on criteria c.
    //! \param id Corresponds to the id of the chosen Nvidia GPU.
    //! \brief 
    __host__
    int	higherComputeCapability(cudaDeviceProp* prop, int id);
  };

};
