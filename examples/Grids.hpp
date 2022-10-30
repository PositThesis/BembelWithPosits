// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef __BEMBEL_GRIDS_
#define __BEMBEL_GRIDS_
#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace Bembel {
namespace Util {

template <typename ptScalar>
inline Eigen::Matrix<ptScalar, Eigen::Dynamic, 3> makeTensorProductGrid(
    Eigen::VectorX<ptScalar> X, Eigen::VectorX<ptScalar> Y, Eigen::VectorX<ptScalar> Z) {
  const int maxX = std::max(X.rows(), X.cols());
  const int maxY = std::max(Y.rows(), Y.cols());
  const int maxZ = std::max(Z.rows(), Z.cols());
  Eigen::Matrix<ptScalar, Eigen::Dynamic, 3> out(maxX * maxY * maxZ, 3);

  for (int iz = 0; iz < maxZ; iz++)
    for (int iy = 0; iy < maxY; iy++)
      for (int ix = 0; ix < maxX; ix++) {
        out.row(ix + iy * maxX + iz * maxY * maxX) =
            Eigen::Vector3<ptScalar>(X(ix), Y(iy), Z(iz));
      }
  return out;
}

template <typename ptScalar>
inline Eigen::Matrix<ptScalar, Eigen::Dynamic, 3> makeSphereGrid(
    const ptScalar r, const int n,
    const Eigen::Vector3<ptScalar> center = Eigen::Vector3<ptScalar>(0, 0, 0)) {
  Eigen::Matrix<ptScalar, Eigen::Dynamic, 3> out(n * n, 3);
  const ptScalar h = 1. / n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      out.row(j + i * n) =
          (Eigen::Vector3<ptScalar>(
               r * cos(3.141592653 * h * i) * sin(3.141592653 * h * (j + 0.5)),
               r * sin(3.141592653 * h * i) * sin(3.141592653 * h * (j + 0.5)),
               r * cos(3.141592653 * h * j)) +
           center);
    }
  }

  return out;
}
}  // namespace Util
}  // namespace Bembel
#endif
