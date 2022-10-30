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
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/Helmholtz>
#include <Bembel/LinearForm>
#include <Bembel/IO>
#include <Eigen/Dense>
#include <iostream>

#include "Data.hpp"
#include "Error.hpp"
#include "Grids.hpp"

using Scalar = sw::universal::posit<64,3>;

int main() {
  using namespace Bembel;
  using namespace Eigen;
  
  Bembel::IO::Stopwatch sw;

  int polynomial_degree_max = 3;
  int refinement_level_max = 3;
  std::complex<Scalar> wavenumber(2., 0.);

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry<Scalar> geometry("sphere.dat");

  // Define evaluation points for scattered field, sphere of radius 2, 10*10
  // points.
  MatrixX<Scalar> gridpoints = Util::makeSphereGrid<Scalar>(2., 10);

  // Define analytical solution using lambda function, in this case the
  // Helmholtz fundamental solution centered on 0, see Data.hpp
  const std::function<std::complex<Scalar>(Vector3<Scalar>)> fun =
      [wavenumber](Vector3<Scalar> pt) {
        return Data::HelmholtzFundamentalSolution(pt, wavenumber,
                                                  Vector3<Scalar>(0., 0., 0.));
      };

  std::cout << "\n" << std::string(60, '=') << "\n";
  // Iterate over polynomial degree.
  for (int polynomial_degree = 0; polynomial_degree < polynomial_degree_max + 1;
       ++polynomial_degree) {
    VectorX<Scalar> error(refinement_level_max + 1);
    // Iterate over refinement levels
    for (int refinement_level = 0; refinement_level < refinement_level_max + 1;
         ++refinement_level) {
      sw.tic();
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level;
      // Build ansatz space
      AnsatzSpace<HelmholtzSingleLayerOperator<Scalar>, Scalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<DirichletTrace<std::complex<Scalar>, Scalar>,
                         HelmholtzSingleLayerOperator<Scalar>, Scalar>
          disc_lf(ansatz_space);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<MatrixX<std::complex<Scalar>>, HelmholtzSingleLayerOperator<Scalar>, Scalar> disc_op(
          ansatz_space);
      disc_op.get_linear_operator().set_wavenumber(wavenumber);
      disc_op.compute();

      // solve system
      PartialPivLU<MatrixX<std::complex<Scalar>>> lu;
      lu.compute(disc_op.get_discrete_operator());
      auto rho = lu.solve(disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<
          HelmholtzSingleLayerPotential<HelmholtzSingleLayerOperator<Scalar>, Scalar>,
          HelmholtzSingleLayerOperator<Scalar>, Scalar>
          disc_pot(ansatz_space);
      disc_pot.get_potential().set_wavenumber(wavenumber);
      disc_pot.set_cauchy_data(rho);
      auto pot = disc_pot.evaluate(gridpoints);

      // compute reference, print time, and compute error
      VectorX<std::complex<Scalar>> pot_ref(gridpoints.rows());
      for (int i = 0; i < gridpoints.rows(); ++i)
        pot_ref(i) = fun(gridpoints.row(i));
      error(refinement_level) = (pot - pot_ref).cwiseAbs().maxCoeff();
      std::cout << " time " << std::setprecision(4) << sw.toc() << "s\t\t";
      std::cout << error(refinement_level) << std::endl;
    }

    // estimate rate of convergence and check whether it is at least 90% of the
    // expected value
    assert(
        checkRateOfConvergence<Scalar>(error.tail(3), 2 * polynomial_degree + 3, 0.9));

    std::cout << std::endl;
  }
  std::cout << std::string(60, '=') << std::endl;

  return 0;
}