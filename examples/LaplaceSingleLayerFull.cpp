// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#include <Eigen/Dense>
#include <universal/number/posit/posit.hpp>
#include <EigenIntegration/Overrides.hpp>
#include <EigenIntegration/std_integration.hpp>

#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include "Data.hpp"
#include "Error.hpp"
#include "Grids.hpp"

#ifdef ScalarIsPosit16
using Scalar = sw::universal::posit<16,2>;
#endif
#ifdef ScalarIsPosit32
using Scalar = sw::universal::posit<32,2>;
#endif
#ifdef ScalarIsPosit64
using Scalar = sw::universal::posit<64,2>;
#endif
#ifdef ScalarIsFloat
using Scalar = float;
#endif
#ifdef ScalarIsDouble
using Scalar = double;
#endif
#ifdef ScalarIsLongDouble
using Scalar = long double;
#endif

int main(int argc, char **argv) {
  using namespace Bembel;
  using namespace Eigen;
  Bembel::IO::Stopwatch sw;

  int polynomial_degree_max = 10;
  int refinement_level_max = 0;

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry<Scalar> geometry("sphere.dat");

  // Define evaluation points for potential field, a tensor product grid of
  // 7*7*7 points in [-.1,.1]^3
  MatrixX<Scalar> gridpoints = Util::makeTensorProductGrid<Scalar>(
      VectorX<Scalar>::LinSpaced(10, -.25, .25), VectorX<Scalar>::LinSpaced(10, -.25, .25),
      VectorX<Scalar>::LinSpaced(10, -.25, .25));

  // Define analytical solution using lambda function, in this case a harmonic
  // function, see Data.hpp
  std::function<Scalar(Vector3<Scalar>)> fun = [](Vector3<Scalar> in) {
    return Data::HarmonicFunction(in);
  };

  assert(argc == 3);
  std::ofstream result(argv[1]);
  result << "time,degree,refinement,error" << std::endl;

  std::ofstream roc(argv[2]);
  roc << "degree,roc" << std::endl;

  std::cerr << "\n" << std::string(60, '=') << "\n";
  // Iterate over polynomial degree.
  for (int polynomial_degree = 1; polynomial_degree < polynomial_degree_max + 1;
       ++polynomial_degree) {
    VectorX<Scalar> error(refinement_level_max + 1);
    // Iterate over refinement levels
    for (int refinement_level = 0; refinement_level < refinement_level_max + 1;
         ++refinement_level) {
      std::cerr << "Poly: " << polynomial_degree << "; Refinement: " << refinement_level << std::endl;
      sw.tic();

      // Build ansatz space
      AnsatzSpace<LaplaceSingleLayerOperator<Scalar>, Scalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      std::cout << "built AnsatzSpace; ";

      // Set up load vector
      DiscreteLinearForm<DirichletTrace<Scalar, Scalar>, LaplaceSingleLayerOperator<Scalar>, Scalar>
          disc_lf(ansatz_space);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      std::cout << "built DLF; ";

      // Set up and compute discrete operator
      DiscreteOperator<MatrixX<Scalar>, LaplaceSingleLayerOperator<Scalar>, Scalar> disc_op(
          ansatz_space);
      disc_op.compute();

      std::cout << "built DiscOp; ";

      // solve system
      LLT<MatrixX<Scalar>> llt;
      llt.compute(disc_op.get_discrete_operator());
      auto rho = llt.solve(disc_lf.get_discrete_linear_form());

      std::cout << "built LLT; ";

      // evaluate potential
      DiscretePotential<LaplaceSingleLayerPotential<LaplaceSingleLayerOperator<Scalar>, Scalar>,
                        LaplaceSingleLayerOperator<Scalar>, Scalar>
          disc_pot(ansatz_space);
      disc_pot.set_cauchy_data(rho);
      auto pot = disc_pot.evaluate(gridpoints);

      std::cout << "built DiscretePotential" << std::endl;;

      // compute reference, print time, and compute error
      VectorX<Scalar> pot_ref(gridpoints.rows());
      for (int i = 0; i < gridpoints.rows(); ++i)
        pot_ref(i) = fun(gridpoints.row(i));
      error(refinement_level) = (pot - pot_ref).cwiseAbs().maxCoeff();
      std::cerr << " time " << std::setprecision(4) << sw.toc() << "s\t\t";
      std::cerr << error(refinement_level) << std::endl;
      result << sw.toc() << "," << polynomial_degree << "," << refinement_level << ","<< error(refinement_level) << std::endl;
    }

    // estimate rate of convergence and check whether it is at least 90% of the
    // expected value
    // assert(
    //     checkRateOfConvergence<Scalar>(error.tail(2), 2 * polynomial_degree + 3, 0.9));

    //roc << polynomial_degree << "," << estimateRateOfConvergence<Scalar>(error.tail(2)) << std::endl;

    std::cerr << std::endl;
  }
  result.close();
  roc.close();
  std::cerr << std::string(60, '=') << std::endl;

  return 0;
}
