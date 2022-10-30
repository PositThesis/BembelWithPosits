// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_TEST_DUFFYTRICK_TESTFUNCTIONS_H_
#define BEMBEL_TEST_DUFFYTRICK_TESTFUNCTIONS_H_

#include <EigenIntegration/Overrides.hpp>
#include <Eigen/Dense>
#include <universal/number/posit/posit.hpp>

namespace Test {
namespace DuffyTrick {

template <typename ptScalar>
ptScalar sigma = 1e-3;

/**
 *  \brief dependent on sigma, this is an almost singular function with an
 *         antiderivative which can easily be computed
 **/
template <typename ptScalar>
ptScalar testFunction1_2D(const Eigen::Vector2<ptScalar> &x) {
  return ptScalar(1.) / (sigma<ptScalar> + BEMBEL_SQUARED_(x(0) - x(1)));
}

template <typename ptScalar>
ptScalar testFuncion1_2DAntiDer(const Eigen::Vector2<ptScalar> &x) {
  return ptScalar(0.5) * std::log(sigma<ptScalar> + BEMBEL_SQUARED_(x(0) - x(1))) +
         (x(1) - x(0)) / sqrt(sigma<ptScalar>) * std::atan((x(0) - x(1)) / sqrt(sigma<ptScalar>));
}
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief analytic integral of the the 2D testfunction wrt a rectangle
 **/
template <typename ptScalar>
ptScalar testFunction2DIntegral(const Eigen::Matrix2<ptScalar> &axis) {
  ptScalar retval = 0;
  Eigen::Vector2<ptScalar> curr_pt;
  Eigen::Vector4<ptScalar> intvals;
  curr_pt << axis(1, 0), axis(1, 1);
  retval += testFuncion1_2DAntiDer(curr_pt);
  intvals(0) = testFuncion1_2DAntiDer(curr_pt);
  curr_pt << axis(1, 0), axis(0, 1);
  retval -= testFuncion1_2DAntiDer(curr_pt);
  intvals(1) = testFuncion1_2DAntiDer(curr_pt);
  curr_pt << axis(0, 0), axis(1, 1);
  retval -= testFuncion1_2DAntiDer(curr_pt);
  intvals(2) = testFuncion1_2DAntiDer(curr_pt);
  curr_pt << axis(0, 0), axis(0, 1);
  retval += testFuncion1_2DAntiDer(curr_pt);
  intvals(3) = testFuncion1_2DAntiDer(curr_pt);
  return retval;
}
////////////////////////////////////////////////////////////////////////////////
// we use the coupling function in 2D to generate a test function
// in 4D which couples to elements
template <typename ptScalar>
ptScalar testFunction4D(const Eigen::Vector2<ptScalar> &x, const Eigen::Vector2<ptScalar> &y) {
  Eigen::Vector2<ptScalar> coupling;
  coupling << x(1), y(0);
  return std::exp(-x(0) - 2 * y(1)) * testFunction1_2D(coupling);
}
// analytic integral of the 4D function
template <typename ptScalar>
ptScalar testFunction4DIntegral(const Eigen::MatrixX<ptScalar> &axis) {
  return (std::exp(-axis(0, 0)) - std::exp(-axis(1, 0))) *
         testFunction2DIntegral<ptScalar>(axis.block(0, 1, 2, 2)) *
         (0.5 * std::exp(-2 * axis(0, 3)) - 0.5 * std::exp(-2 * axis(1, 3)));
}
////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief this function uses a simple tensor product quadrature, we test
 *  it with an analytic function
 **/
template <typename ptScalar>
std::function<ptScalar(const Eigen::Vector2<ptScalar> &, const Eigen::Vector2<ptScalar> &)>
    integrate0_test_function =
        [](const Eigen::Vector2<ptScalar> &x, const Eigen::Vector2<ptScalar> &y) {
          return std::sin(BEMBEL_PI * x(0)) * std::sin(2 * BEMBEL_PI * y(0)) *
                 std::exp(-x(1)) * std::exp(y(1));
        };
template <typename ptScalar>
std::function<ptScalar(const Eigen::MatrixX<ptScalar> &)>
    integrate0_test_function_integral = [](const Eigen::MatrixX<ptScalar> &axis) {
      return (std::cos(BEMBEL_PI * axis(0, 0)) -
              std::cos(BEMBEL_PI * axis(1, 0))) /
             BEMBEL_PI * 0.5 *
             (std::cos(2 * BEMBEL_PI * axis(0, 2)) -
              std::cos(2 * BEMBEL_PI * axis(1, 2))) /
             BEMBEL_PI * (std::exp(-axis(0, 1)) - std::exp(-axis(1, 1))) *
             (std::exp(axis(1, 3)) - std::exp(axis(0, 3)));
    };
/**
 *  \brief this function uses a simple tensor product quadrature, we test
 *  it with an analytic function
 **/
template <typename ptScalar>
std::function<ptScalar(const Eigen::Vector2<ptScalar> &, const Eigen::Vector2<ptScalar> &)>
    integrate1_test_function =
        [](const Eigen::Vector2<ptScalar> &x, const Eigen::Vector2<ptScalar> &y) {
          return std::sin(BEMBEL_PI * x(0)) * std::sin(2 * BEMBEL_PI * y(0)) *
                 std::exp(-x(1)) * std::exp(y(1));
        };
template <typename ptScalar>
std::function<ptScalar(const Eigen::MatrixX<ptScalar> &)>
    integrate1_test_function_integral = [](const Eigen::MatrixX<ptScalar> &axis) {
      return (std::cos(BEMBEL_PI * axis(0, 0)) -
              std::cos(BEMBEL_PI * axis(1, 0))) /
             BEMBEL_PI * 0.5 *
             (std::cos(2 * BEMBEL_PI * axis(0, 2)) -
              std::cos(2 * BEMBEL_PI * axis(1, 2))) /
             BEMBEL_PI * (std::exp(-axis(0, 1)) - std::exp(-axis(1, 1))) *
             (std::exp(axis(1, 3)) - std::exp(axis(0, 3)));
    };

template <typename ptScalar>
std::function<ptScalar(const Eigen::Vector2<ptScalar> &, const Eigen::Vector2<ptScalar> &)>
    integrate2_test_function =
        [](const Eigen::Vector2<ptScalar> &x, const Eigen::Vector2<ptScalar> &y) {
#ifdef BEMBEL_TEST_DUFFYTRICK_USE_ANALYTIC_FUNCTION_
          return std::sin(BEMBEL_PI * x(0)) * std::sin(2 * BEMBEL_PI * y(0)) *
                 std::exp(-x(1)) * std::exp(y(1));
#else
          return testFunction4D(x, y);
#endif
        };
template <typename ptScalar>
std::function<ptScalar(const Eigen::MatrixX<ptScalar> &)>
    integrate2_test_function_integral = [](const Eigen::MatrixX<ptScalar> &axis) {
#ifdef BEMBEL_TEST_DUFFYTRICK_USE_ANALYTIC_FUNCTION_
      return (std::cos(BEMBEL_PI * axis(0, 0)) -
              std::cos(BEMBEL_PI * axis(1, 0))) /
             BEMBEL_PI * 0.5 *
             (std::cos(2 * BEMBEL_PI * axis(0, 2)) -
              std::cos(2 * BEMBEL_PI * axis(1, 2))) /
             BEMBEL_PI * (std::exp(-axis(0, 1)) - std::exp(-axis(1, 1))) *
             (std::exp(axis(1, 3)) - std::exp(axis(0, 3)));
#else
      return testFunction4DIntegral(axis);
#endif
    };

template <typename ptScalar>
std::function<ptScalar(const Eigen::Vector2<ptScalar> &, const Eigen::Vector2<ptScalar> &)>
    integrate3_test_function =
        [](const Eigen::Vector2<ptScalar> &x, const Eigen::Vector2<ptScalar> &y) {
#ifdef BEMBEL_TEST_DUFFYTRICK_USE_ANALYTIC_FUNCTION_
          return std::sin(BEMBEL_PI * x(0)) * std::sin(2 * BEMBEL_PI * y(0)) *
                 std::exp(-x(1)) * std::exp(y(1));
#else
          return testFunction4D(x, y);
#endif
        };
template <typename ptScalar>
std::function<ptScalar(const Eigen::MatrixX<ptScalar> &)>
    integrate3_test_function_integral = [](const Eigen::MatrixX<ptScalar> &axis) {
#ifdef BEMBEL_TEST_DUFFYTRICK_USE_ANALYTIC_FUNCTION_
      return (std::cos(BEMBEL_PI * axis(0, 0)) -
              std::cos(BEMBEL_PI * axis(1, 0))) /
             BEMBEL_PI * 0.5 *
             (std::cos(2 * BEMBEL_PI * axis(0, 2)) -
              std::cos(2 * BEMBEL_PI * axis(1, 2))) /
             BEMBEL_PI * (std::exp(-axis(0, 1)) - std::exp(-axis(1, 1))) *
             (std::exp(axis(1, 3)) - std::exp(axis(0, 3)));
#else
      return testFunction4DIntegral(axis);
#endif
    };

template <typename ptScalar>
std::function<ptScalar(const Eigen::Vector2<ptScalar> &, const Eigen::Vector2<ptScalar> &)>
    integrate4_test_function =
        [](const Eigen::Vector2<ptScalar> &x, const Eigen::Vector2<ptScalar> &y) {
#ifdef BEMBEL_TEST_DUFFYTRICK_USE_ANALYTIC_FUNCTION_
          return std::sin(BEMBEL_PI * x(0)) * std::sin(2 * BEMBEL_PI * y(0)) *
                 std::exp(-x(1)) * std::exp(y(1));
#else
          return testFunction4D(x, y);
#endif
        };
template <typename ptScalar>
std::function<ptScalar(const Eigen::MatrixX<ptScalar> &)>
    integrate4_test_function_integral = [](const Eigen::MatrixX<ptScalar> &axis) {
#ifdef BEMBEL_TEST_DUFFYTRICK_USE_ANALYTIC_FUNCTION_
      return (std::cos(BEMBEL_PI * axis(0, 0)) -
              std::cos(BEMBEL_PI * axis(1, 0))) /
             BEMBEL_PI * 0.5 *
             (std::cos(2 * BEMBEL_PI * axis(0, 2)) -
              std::cos(2 * BEMBEL_PI * axis(1, 2))) /
             BEMBEL_PI * (std::exp(-axis(0, 1)) - std::exp(-axis(1, 1))) *
             (std::exp(axis(1, 3)) - std::exp(axis(0, 3)));
#else
      return testFunction4DIntegral(axis);
#endif
    };
}  // namespace DuffyTrick
}  // namespace Test
#endif
