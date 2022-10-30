// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#include <EigenIntegration/Overrides.hpp>
#include <Eigen/Dense>
#include <universal/number/posit/posit.hpp>

#include <Bembel/AnsatzSpace>
#include <Bembel/DummyOperator>
#include <Bembel/Geometry>

#include "Test.hpp"
#include "TestGeometries.hpp"

using Scalar = sw::universal::posit<64, 3>;

int main() {
  using namespace Bembel;
  using namespace Eigen;

  int number_of_points = 9;
  int number_of_points2 = number_of_points * number_of_points;
  int refinement_level = 3;
  int cluster_level = 1;

  // generate geometry
  Test::TestGeometryWriter::writeScreen();
  Geometry<Scalar> geometry("test_Screen.dat");
  AnsatzSpace<DummyOperator<Scalar>, Scalar> ansatz_space(geometry, refinement_level, 0);

  // generate matrices
  MatrixX<Scalar> fmm_transfer_matrices =
      H2Multipole::computeTransferMatrices<H2Multipole::ChebychevRoots<Scalar>, Scalar>(
          number_of_points);
  std::vector<MatrixX<Scalar>> fmm_moment_matrix1 =
      H2Multipole::Moment2D<H2Multipole::ChebychevRoots<Scalar>, DummyOperator<Scalar>, Scalar>::
          compute2DMoment(ansatz_space.get_superspace(), cluster_level,
                          refinement_level - cluster_level, number_of_points);
  std::vector<MatrixX<Scalar>> fmm_moment_matrix2 = H2Multipole::
      Moment2D<H2Multipole::ChebychevRoots<Scalar>, DummyOperator<Scalar>, Scalar>::compute2DMoment(
          ansatz_space.get_superspace(), cluster_level - 1,
          refinement_level - cluster_level + 1, number_of_points);

  // apply tranfer matrices
  MatrixX<Scalar> result0 =
      fmm_moment_matrix2[0].block(0, 0 * fmm_moment_matrix1[0].cols(),
                                  fmm_moment_matrix1[0].rows(),
                                  fmm_moment_matrix1[0].cols()) -
      fmm_transfer_matrices.block(0, 0 * number_of_points2, number_of_points2,
                                  number_of_points2) *
          fmm_moment_matrix1[0];
  MatrixX<Scalar> result1 =
      fmm_moment_matrix2[0].block(0, 1 * fmm_moment_matrix1[0].cols(),
                                  fmm_moment_matrix1[0].rows(),
                                  fmm_moment_matrix1[0].cols()) -
      fmm_transfer_matrices.block(0, 1 * number_of_points2, number_of_points2,
                                  number_of_points2) *
          fmm_moment_matrix1[0];
  MatrixX<Scalar> result2 =
      fmm_moment_matrix2[0].block(0, 2 * fmm_moment_matrix1[0].cols(),
                                  fmm_moment_matrix1[0].rows(),
                                  fmm_moment_matrix1[0].cols()) -
      fmm_transfer_matrices.block(0, 2 * number_of_points2, number_of_points2,
                                  number_of_points2) *
          fmm_moment_matrix1[0];
  MatrixX<Scalar> result3 =
      fmm_moment_matrix2[0].block(0, 3 * fmm_moment_matrix1[0].cols(),
                                  fmm_moment_matrix1[0].rows(),
                                  fmm_moment_matrix1[0].cols()) -
      fmm_transfer_matrices.block(0, 3 * number_of_points2, number_of_points2,
                                  number_of_points2) *
          fmm_moment_matrix1[0];

  // test for correctness, check for each norm separately if test fails
  BEMBEL_TEST_IF(std::abs(result0.norm() + result1.norm() + result2.norm() +
                          result3.norm()) < 1e-12);
}
