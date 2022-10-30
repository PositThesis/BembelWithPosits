// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
//


#include <EigenIntegration/Overrides.hpp>
#include <Eigen/Dense>
#include <universal/number/posit/posit.hpp>

#include <Bembel/AnsatzSpace>
#include <Bembel/Geometry>
#include <Bembel/H2Matrix>
#include <Bembel/IO>
#include <Bembel/Laplace>
#include <Bembel/LinearForm>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <iostream>

#include "Data.hpp"
#include "Error.hpp"
#include "Grids.hpp"


using Scalar = sw::universal::posit<64, 3>;

int main() {
  using namespace Bembel;
  using namespace Eigen;

  int polynomial_degree_max = 3;
  int refinement_level_max = 3;

  // Load geometry from file "sphere.dat", which must be placed in the same
  // directory as the executable
  Geometry<Scalar> geometry("torus.dat");

  // Define evaluation points for potential field, a tensor product grid of
  // 7*7*7 points in [-.1,.1]^3
  MatrixX<Scalar> gridpoints = Util::makeTensorProductGrid<Scalar>(
      VectorX<Scalar>::LinSpaced(10, -.1, .1), VectorX<Scalar>::LinSpaced(10, -2.1, -1.9),
      VectorX<Scalar>::LinSpaced(10, -.1, .1));

  // Define analytical solution using lambda function, in this case a harmonic
  // function, see Data.hpp
  std::function<Scalar(Vector3<Scalar>)> fun = [](Vector3<Scalar> in) {
    return Data::HarmonicFunction(in);
  };

  // Iterate over polynomial degree.
  for (int polynomial_degree = 0; polynomial_degree < polynomial_degree_max + 1;
       ++polynomial_degree) {
    VectorX<Scalar> error(refinement_level_max + 1);
    IO::Logger<12> logger("log_LaplaceSingle_" +
                          std::to_string(polynomial_degree) + ".log");
    logger.both("P", "M", "error");
    // Iterate over refinement levels
    for (int refinement_level = 0; refinement_level < refinement_level_max + 1;
         ++refinement_level) {
      std::cout << "Degree " << polynomial_degree << " Level "
                << refinement_level << "\t\t";
      // Build ansatz space
      AnsatzSpace<LaplaceSingleLayerOperator<Scalar>, Scalar> ansatz_space(
          geometry, refinement_level, polynomial_degree);

      // Set up load vector
      DiscreteLinearForm<DirichletTrace<Scalar, Scalar>, LaplaceSingleLayerOperator<Scalar>, Scalar>
          disc_lf(ansatz_space);
      disc_lf.get_linear_form().set_function(fun);
      disc_lf.compute();

      // Set up and compute discrete operator
      DiscreteOperator<H2Matrix<Scalar, Scalar>, LaplaceSingleLayerOperator<Scalar>, Scalar> disc_op(
          ansatz_space);
      disc_op.compute();

      // solve system
      ConjugateGradient<H2Matrix<Scalar, Scalar>, Lower | Upper, IdentityPreconditioner>
          cg;
      cg.compute(disc_op.get_discrete_operator());
      auto rho = cg.solve(disc_lf.get_discrete_linear_form());

      // evaluate potential
      DiscretePotential<LaplaceSingleLayerPotential<LaplaceSingleLayerOperator<Scalar>, Scalar>,
                        LaplaceSingleLayerOperator<Scalar>, Scalar>
          disc_pot(ansatz_space);
      disc_pot.set_cauchy_data(rho);
      auto pot = disc_pot.evaluate(gridpoints);

      // compute reference and compute error
      VectorX<Scalar> pot_ref(gridpoints.rows());
      for (int i = 0; i < gridpoints.rows(); ++i)
        pot_ref(i) = fun(gridpoints.row(i));
      error(refinement_level) = (pot - pot_ref).cwiseAbs().maxCoeff();

      logger.both(polynomial_degree, refinement_level, error(refinement_level));

      // we only need one visualization
      if (refinement_level == 3 && polynomial_degree == 2) {
        VTKSurfaceExport<Scalar> writer(geometry, 5);

        FunctionEvaluator<LaplaceSingleLayerOperator<Scalar>, Scalar> evaluator(ansatz_space);
        evaluator.set_function(rho);

        std::function<Scalar(int, const Eigen::Vector2<Scalar> &)> density =
            [&](int patch_number,
                const Eigen::Vector2<Scalar> &reference_domain_point) {
              return evaluator.evaluateOnPatch(patch_number,
                                               reference_domain_point)(0);
            };
        writer.addDataSet("Density", density);
        writer.writeToFile("LaplaceSingle.vtp");
      }
    }

    std::cout << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl;

  return 0;
}
