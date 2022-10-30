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
#include <iostream>

#include "Bembel/src/AnsatzSpace/FunctionEvaluator.hpp"

using Scalar = sw::universal::posit<64,3>;

int main() {
  int refinement_level = 4;
  int polynomial_degree = 0;
  Bembel::Geometry<Scalar> geometry("sphere.dat");
  std::cout << "The geometry has " << geometry.get_number_of_patches()
            << " patches." << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  Bembel::AnsatzSpace<Bembel::DummyOperator<Scalar>, Scalar>(geometry, refinement_level,
                                             polynomial_degree);
  std::cout
      << "The operator order is "
      << Bembel::LinearOperatorTraits<Bembel::DummyOperator<Scalar>>::OperatorOrder
      << "." << std::endl;
  return 0;
}