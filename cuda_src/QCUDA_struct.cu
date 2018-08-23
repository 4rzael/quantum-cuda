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


template<typename T> __host__ __device__
QCUDA::s_complex<T>::s_complex()
  : real_(0), imag_(0)
{}


template<typename T> __host__ __device__
QCUDA::s_complex<T>::s_complex(T r, T i)
  : real_(r), imag_(i)
{}  


template<typename T> __host__ __device__
QCUDA::s_complex<T>::s_complex(const struct s_complex& cpy)
  : real_(cpy.real_), imag_(cpy.imag_)
{}


template<typename T> __host__ __device__
QCUDA::s_complex<T>::~s_complex() = default;


template<typename T> __host__ __device__
struct QCUDA::s_complex<T>&	QCUDA::s_complex<T>::operator=(const struct s_complex& cpy) {
  this->real_ = cpy.real_;
  this->imag_ = cpy.imag_;
  return (*this);
}


template<typename T> __host__ __device__
T QCUDA::s_complex<T>::norm() {
  return real_ * real_ + imag_ * imag_;
}


template<typename T> __host__ __device__
QCUDA::s_complex<T> QCUDA::s_complex<T>::operator/(const T &v) {
  return QCUDA::s_complex<T>(real_ / v, imag_ / v);
}


template<typename T> __host__ __device__
struct QCUDA::s_complex<T>	QCUDA::s_complex<T>::operator+(const struct s_complex<T>& c) {
  struct s_complex<T> tmp;

  tmp.real_ = this->real_ + c.real_;
  tmp.imag_ = this->imag_ + c.imag_;
  return (tmp);
}


template<typename T> __host__ __device__
struct QCUDA::s_complex<T>	QCUDA::s_complex<T>::operator*(const struct s_complex<T>& c) {
  struct s_complex<T> tmp;

  tmp.real_ = (this->real_ * c.real_) + (this->imag_ * c.imag_ * -1);
  tmp.imag_ = (this->real_ * c.imag_) + (this->imag_ * c.real_);
  return (tmp);
}


template<typename T> __host__ __device__
struct QCUDA::s_complex<T>&	QCUDA::s_complex<T>::operator+=(const struct s_complex<T>& c) {
  this->real_ += c.real_;
  this->imag_ += c.imag_;
  return (*this);
}


template<typename T> __host__
std::ostream&	operator<<(std::ostream& os, const struct QCUDA::s_complex<T>& c) {
  os << "(" << c.real_ << ", " << c.imag_ << "i)" << std::endl;
  return (os);
}


//!
//! Since templates are defined outside of the header file, due to the fact that
//! CUDA source codes must be defined within a '.cu' file, therefore we have to
//! explicitly instantiate all the types that can be handled by the structure
//! 'struct s_complex'.Hence 'float' and 'double' types.
//!
template struct QCUDA::s_complex<double>;

template struct QCUDA::s_complex<float>;
