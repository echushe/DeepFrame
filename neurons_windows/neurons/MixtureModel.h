#pragma once
#include "TMatrix.h"
#include <iostream>

namespace neurons
{
    struct EM_Gaussian_1d
    {
        double m_mu;
        double m_sigma;
        double m_probability;

        EM_Gaussian_1d();
        EM_Gaussian_1d(double mu, double sigma, double probability);
    };

    class EM_1d
    {
    private:
        lint m_gssns;

    public:
        EM_1d();
        EM_1d(lint gaussians);

    public:
        std::vector<EM_Gaussian_1d> operator ()(
            std::vector<neurons::TMatrix<>> &p_x_in_gaussians,
            std::vector<neurons::TMatrix<>> &p_gaussians_in_x,
            const neurons::TMatrix<> & input);

        ~EM_1d();
    };
}

