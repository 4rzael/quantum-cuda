/**
 * @Author: Nicolas Jankovic <nj203>
 * @Date:   2018-06-16T10:08:10+01:00
 * @Email:  nicolas.jankovic@epitech.eu
 * @Project: CUDA-Based Simulator of Quantum Systems
 * @Filename: QCUDA_struct.cuh
 * @Last modified by:   nj203
 * @Last modified time: 2018-06-27T14:51:52+01:00
 * @License: MIT License
 */

# include "QCUDA_struct.cuh"

/**
 * This file contains all the definitions of the structure 'struct s_complex'.
 */

/**
 * Each instance of the structure 'struct s_complex' will initialize
 * the real part 'real_' and the imaginary part 'imag_' to 0, thanks to the
 * initializer list.
 */
template<typename T> __host__ __device__
QCUDA::s_complex<T>::s_complex()
  : real_(0), imag_(0)
{}


template<typename T> __host__ __device__
QCUDA::s_complex<T>::~s_complex()
{}


template<typename T> __host__ __device__
T	QCUDA::s_complex<T>::getReal() {
  return (this->real_);
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::setReal(const T& v) {
  this->real_ = v;
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::aggregateReal(const T& v) {
  this->real_ += v;
}


template<typename T> __host__ __device__
T	QCUDA::s_complex<T>::getImag() {
  return (this->imag_);
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::setImag(const T& v) {
  this->imag_ = v;
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::aggregateImag(const T& v) {
  this->imag_ += v;
}


template<typename T> __host__ __device__
void	QCUDA::s_complex<T>::operator+=(const struct s_complex<T>& c) {
  this->real_ += c.real_;
  this->imag_ += c.imag_;
}


template<typename T> __host__ __device__
struct QCUDA::s_complex<T>	QCUDA::s_complex<T>::operator+(const struct s_complex<T>& c) {
  struct s_complex<T> tmp;

  tmp.real_ = this->real_ + c.real_;
  tmp.imag_ = this->imag_ + c.imag_;
  return (tmp);
}


template<typename T> __host__
std::ostream&	operator<<(std::ostream& os, const struct QCUDA::s_complex<T>& c) {
  os << c.real_ << std::endl;
  os << c.imag_ << std::endl;
  return (os);
}

/**
 * Since templates are defined outside of the header file, due to the fact that
 * CUDA source codes must be defined within a '.cu' file, therefore we have to
 * explicitly instantiate all the types that can be handled by the structure
 * 'struct s_complex'.Hence 'float' and 'double' types.
 */
template struct QCUDA::s_complex<double>;
template struct QCUDA::s_complex<float>;
