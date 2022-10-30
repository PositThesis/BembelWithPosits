// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef __BEMBEL_DATA_
#define __BEMBEL_DATA_

#include <Eigen/Dense>
#include <cmath>

namespace Bembel {
namespace Data {

/*	@brief This function implements a harmonic function, which, in the
 * interior dormain, satisfies the Laplace equation.
 *
 */
template <typename ptScalar>
inline ptScalar HarmonicFunction(Eigen::Vector3<ptScalar> in) {
  return (4 * in(0) * in(0) - 3 * in(1) * in(1) - in(2) * in(2));
}

/*	@brief This function implements the Helmholtz fundamental solution,
 * which, if center is placed in the interior domain, satisfies the Helmholtz
 * equation and radiation condition in the exterior domain.
 *
 * Note that the sign in the fundamental solution is linked to the sign of kappa
 * in this function.
 */
template <typename ptScalar>
inline std::complex<ptScalar> HelmholtzFundamentalSolution(
    Eigen::Vector3<ptScalar> pt, std::complex<ptScalar> kappa,
    Eigen::Vector3<ptScalar> center = Eigen::Vector3<ptScalar>(0, 0, 0)) {
  return std::exp(-std::complex<ptScalar>(0, 1) * kappa * (pt - center).norm()) /
         (pt - center).norm();
}

/* @brief This function implement a Hertz Dipole as in page 411 of J.D.Jacksons
 * "Classical Electrodynamics", 3rd ed., which, if the dipole axis given by
 * position and length remains in the interior domain, satisfies the curl-curl
 * equation and the radiation condition in the exterior domain.
 *
 * Note that the sign in the fundamental solution is linked to the sign of kappa
 * in this function.
 */
template <typename ptScalar>
inline Eigen::Vector3<std::complex<ptScalar>> Dipole(Eigen::Vector3<ptScalar> pt, std::complex<ptScalar> kappa,
                               Eigen::Vector3<ptScalar> position,
                               Eigen::Vector3<ptScalar> length) {
  std::complex<ptScalar> i(0, 1);
  const Eigen::Vector3<ptScalar> c = pt - position;
  ptScalar r = c.norm();
  const Eigen::Vector3<ptScalar> n = c / r;
  const std::complex<ptScalar> expc = std::exp(-i * kappa * r);
  const Eigen::Vector3<std::complex<ptScalar>> E =
      (kappa * kappa * expc / r) * (n.cross(length).cross(n));
  const Eigen::Vector3<ptScalar> h = 3 * n.dot(length) * n - length;
  const std::complex<ptScalar> ec =
      ((1 / (r * r * r)) - (i * (-kappa) / (r * r))) * expc;
  return E + ec * h;
}

}  // namespace Data
}  // namespace Bembel
#endif
