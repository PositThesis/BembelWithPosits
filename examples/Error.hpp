// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef __BEMBEL_UTIL_ERROR__
#define __BEMBEL_UTIL_ERROR__

/**
 * @brief Routines for the evalutation of pointwise errors.
 */

namespace Bembel {

template <typename Scalar, typename ptScalar>
inline Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> errors(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &pot,
    const Eigen::MatrixX<ptScalar> &grid,
    const std::function<Scalar(Eigen::Vector3d)> &fun) {
  const int gridsz = grid.rows();
  assert((std::max(pot.cols(), pot.rows()) == grid.rows()) &&
         ("The size does not match!"));
  assert((grid.cols() == 3) &&
         "The grid must be a Matrix with a 3d point in each row!");
  assert((std::min(pot.rows(), pot.cols())) && ("Potential must be Vector!"));

  Eigen::Matrix<ptScalar, 1, Eigen::Dynamic> errors(gridsz);

  for (int i = 0; i < gridsz; i++) {
    errors(i) = std::abs(pot(i) - fun((grid.row(i).transpose()).eval()));
  }
  return errors;
}
template <typename ptScalar>
inline Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> errors(
    const Eigen::MatrixXcd &pot, const Eigen::MatrixX<ptScalar> &grid,
    const std::function<Eigen::Vector3cd(Eigen::Vector3d, std::complex<ptScalar>)>
        &fun,
    std::complex<ptScalar> kappa) {
  const int gridsz = grid.rows();
  assert((std::max(pot.cols(), pot.rows()) == grid.rows()) &&
         ("The size does not match!"));
  assert((grid.cols() == 3) &&
         "The grid must be a Matrix with a 3d point in each row!");
  assert((pot.cols() == 3) &&
         ("Potential must be a Matrix with a Vector3cd in each col!"));
  Eigen::Matrix<ptScalar, 1, Eigen::Dynamic> errors(gridsz);

  for (int i = 0; i < gridsz; i++) {
    errors(i) =
        (pot.col(i) - fun((grid.row(i).transpose()).eval(), kappa)).norm();
  }
  return errors;
}

template <typename Scalar, typename ptScalar>
inline ptScalar maxPointwiseError(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &pot,
    const Eigen::MatrixX<ptScalar> &grid,
    const std::function<Scalar(Eigen::Vector3d)> &fun) {
  const int gridsz = grid.rows();
  assert((std::max(pot.cols(), pot.rows()) == grid.rows()) &&
         ("The size does not match!"));
  assert((grid.cols() == 3) &&
         "The grid must be a Matrix with a 3d point in each row!");
  assert((std::min(pot.cols(), pot.rows()) == 1) &&
         ("Potential must be a vector!"));
  ptScalar error = 0;

  for (int i = 0; i < gridsz; i++) {
    ptScalar tmp = std::abs(pot(i) - fun((grid.row(i).transpose()).eval()));
    error = tmp > error ? tmp : error;
  }
  return error;
}
template <typename ptScalar>
inline ptScalar maxPointwiseError(
    const Eigen::MatrixXcd &pot, const Eigen::MatrixX<ptScalar> &grid,
    const std::function<Eigen::Vector3cd(Eigen::Vector3d)> &fun) {
  const int gridsz = grid.rows();
  assert(pot.cols() == grid.cols());
  assert((std::max(pot.cols(), pot.rows()) == grid.rows()) &&
         ("The size does not match!"));
  assert((grid.cols() == 3) &&
         "The grid must be a Matrix with a 3d point in each row!");
  assert((pot.cols() == 3) &&
         ("Must be a Matrix with a point solution in each row!"));
  ptScalar error = 0;

  for (int i = 0; i < gridsz; i++) {
    ptScalar tmp = (pot.row(i) - fun(grid.row(i)).transpose()).norm();
    error = tmp > error ? tmp : error;
  }
  return error;
}

template <typename ptScalar>
ptScalar estimateRateOfConvergence(const Eigen::VectorX<ptScalar> &errors) {
  Eigen::MatrixX<ptScalar> A(errors.rows(), 2);
  A << Eigen::VectorX<ptScalar>::Ones(errors.rows()),
      Eigen::VectorX<ptScalar>::LinSpaced(errors.rows(), 0, errors.rows() - 1);
  Eigen::VectorX<ptScalar> b = errors.array().abs().log() / std::log(2);
  Eigen::VectorX<ptScalar> x = A.colPivHouseholderQr().solve(b);
  return -x(1);
}

template <typename ptScalar>
bool checkRateOfConvergence(const Eigen::VectorX<ptScalar> &errors,
                            const int expected_rate, const ptScalar tol_factor,
                            ptScalar *rate_of_convergence_out = NULL) {
  ptScalar rate_of_convergence = estimateRateOfConvergence(errors);
  if (rate_of_convergence_out) rate_of_convergence_out[0] = rate_of_convergence;
  std::cout << "Estimated rate of convergence:" << rate_of_convergence
            << std::endl;
  return (rate_of_convergence > tol_factor * expected_rate);
}

}  // namespace Bembel

#endif
