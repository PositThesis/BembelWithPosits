#include <Bembel/Quadrature>
#include <iostream>

#include <EigenIntegration/Overrides.hpp>
#include <Eigen/Dense>
#include <universal/number/posit/posit.hpp>

using Scalar = sw::universal::posit<64,3>;

void computeLegendreQuadrature(Eigen::VectorX<Scalar> *xi, Eigen::VectorX<Scalar> *w,
                               int order) {
  Eigen::MatrixX<Scalar> A(order, order);
  A.setZero();
  for (auto i = 1; i < order; ++i) A(i, i - 1) = i / (sqrt(4. * i * i - 1));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> es;
  es.compute(A);
  xi->resize(order);
  w->resize(order);
  for (auto i = 0; i < order; ++i) {
    (*xi)(i) = 0.5 * (es.eigenvalues()(i) + 1);
    (*w)(i) = es.eigenvectors()(0, i) * es.eigenvectors()(0, i);
  }
  return;
}

Eigen::VectorX<Scalar> HornersMethod(const Eigen::VectorX<Scalar> &c,
                              const Eigen::VectorX<Scalar> &x) {
  Eigen::VectorX<Scalar> retval(x.size());
  retval.array() = c(0);
  for (auto i = 1; i < c.size(); ++i) {
    retval.array() = x.cwiseProduct(retval).array() + c(i);
  }
  return retval;
}

int main() {
  constexpr unsigned int max_order = 100;
  constexpr unsigned int max_tp_order = 100;
  Eigen::VectorX<Scalar> xi;
  Eigen::VectorX<Scalar> w;

  Bembel::GaussLegendre<max_order, Scalar> GS;
  // compare GaussLegendre rules versus nodes and weights obtained from
  // solving the eigenvalue problem
  {
    for (auto i = 0; i < max_order; ++i) {
      computeLegendreQuadrature(&xi, &w, i + 1);
      std::cout << "order: " << i + 1 << std::endl;
      std::cout << (xi - GS[i].xi_.transpose()).norm() / (GS[i].xi_).norm()
                << std::endl;
      std::cout << (w - GS[i].w_).norm() / (GS[i].w_).norm() << std::endl;
      std::cout << "---------------\n";
    }
  }
  // perform an actual quadrature test;
  {
    for (auto i = 0; i < max_order; ++i) {
      Eigen::VectorX<Scalar> c = Eigen::VectorX<Scalar>::Random(2 * i + 2);
      Eigen::VectorX<Scalar> integral_weights =
          1. / Eigen::ArrayX<Scalar>::LinSpaced(2 * i + 2, 1, 2 * i + 2).reverse();
      Scalar exact_integral = c.cwiseProduct(integral_weights).sum();
      std::cout << "degree of test polynomial: " << 2 * i + 1
                << " integral: " << exact_integral << std::endl;
      for (auto j = 0; j < max_order; ++j) {
        Eigen::VectorX<Scalar> pol_eval = HornersMethod(c, GS[j].xi_);
        Scalar quadrature_val = (pol_eval.array() * GS[j].w_.array()).sum();
        std::cout << "quadrature degree (i.e. #qpoints) " << j + 1 << " error: "
                  << std::abs(quadrature_val - exact_integral) /
                         std::abs(exact_integral)
                  << std::endl;
      }
      std::cout << "---------------\n";
    }

    // tensor product quadrature
    {
      Scalar exact_integral = (std::cos(BEMBEL_PI) - std::cos(0)) / BEMBEL_PI;
      exact_integral *= exact_integral;

      Bembel::GaussSquare<max_tp_order, Scalar> GS;
      for (auto j = 0; j < max_tp_order; ++j) {
        Scalar quadrature_val = 0;
        for (auto i = 0; i < GS[j].xi_.cols(); ++i)
          quadrature_val += GS[j].w_(i) * std::sin(BEMBEL_PI * GS[j].xi_(0, i)) *
                            std::sin(BEMBEL_PI * GS[j].xi_(1, i));
        std::cout << "TP quadrature degree " << j + 1 << " error: "
                  << std::abs(quadrature_val - exact_integral) /
                         std::abs(exact_integral)
                  << std::endl;
      }
    }
  }
  return 0;
}
