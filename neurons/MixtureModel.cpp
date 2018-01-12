#include "MixtureModel.h"
#include "Functions.h"

neurons::EM_Gaussian_1d::EM_Gaussian_1d()
    : m_mu{ 0 }, m_sigma{ 0.1 }, m_probability{ 0.1 }
{}


neurons::EM_Gaussian_1d::EM_Gaussian_1d(double mu, double sigma, double probability)
    : m_mu{ mu }, m_sigma{ sigma }, m_probability{ probability }
{}


neurons::EM_1d::EM_1d()
    : m_gssns{ 0 }
{}


neurons::EM_1d::EM_1d(lint gaussians)
    : m_gssns{ gaussians }
{}


std::vector<neurons::EM_Gaussian_1d> neurons::EM_1d::operator()(
    std::vector<neurons::Matrix> &p_x_in_gaussians,
    std::vector<neurons::Matrix> &p_gaussians_in_x,
    const neurons::Matrix & input)
{
    lint input_size = input.shape().size();
    double max = input.max();
    double min = input.min();
    std::cout << "min = " << min << '\n';
    std::cout << "max = " << max << '\n';
    double range = max - min;
    double full_prob = 1;

    for (lint i = 0; i < this->m_gssns; ++i)
    {
        p_gaussians_in_x.push_back(Matrix{ input.shape() });
        p_x_in_gaussians.push_back(Matrix{ input.shape() });
    }

    // Step1
    // 1. assume the probability of each gaussian distribution is 1 / m_gssns
    // 2. assume mean of each gaussian distribution is evenly distributed from min to max
    // 3. assume variance of each gaussian is ((max - min) / gaussians)
    std::vector<EM_Gaussian_1d> gaussians{ static_cast<size_t>(this->m_gssns) };
    for (lint i = 0; i < this->m_gssns; ++i)
    {
        gaussians[i].m_mu = min + range * (i + 0.5) / this->m_gssns;
        gaussians[i].m_sigma = range / this->m_gssns;
        gaussians[i].m_probability = full_prob / this->m_gssns;
    }

    double *bayes_dividers = new double[input_size];

    // Re-do step2, 3, 4, 5
    for (lint step  = 0; step < 200; ++step)
    {
        /*
        for (size_t i = 0; i < gaussians.size(); ++i)
        {
            std::cout << "Gaussian " << i << ": mu = " << gaussians[i].m_mu
                << ", sigma = " << gaussians[i].m_sigma << ", probability = " << gaussians[i].m_probability << '\n';

            std::cout << p_x_in_gaussians[i];
            std::cout << p_gaussians_in_x[i];
            std::cout << '\n';
        }
        */

        // Step2 
        // Calculate p_x_in_gaussians
        for (lint i = 0; i < this->m_gssns; ++i)
        {
            for (lint j = 0; j < input_size; ++j)
            {
                p_x_in_gaussians[i].m_data[j] = 
                    gaussian_function(gaussians[i].m_mu, gaussians[i].m_sigma, input.m_data[j]);
            } 
        }

        // Step3
        // calculate bayes dividers
        for (lint i = 0; i < input_size; ++i)
        {
            double sum = 0;
            for (lint j = 0; j < this->m_gssns; ++j)
            {
                sum += p_x_in_gaussians[j].m_data[i] * gaussians[j].m_probability;
            }
            bayes_dividers[i] = sum;
        }

        // Step4
        // calculate p_gaussians_in_x
        for (lint i = 0; i < this->m_gssns; ++i)
        {
            for (lint j = 0; j < input_size; ++j)
            {
                p_gaussians_in_x[i].m_data[j] = 
                    p_x_in_gaussians[i].m_data[j] * gaussians[i].m_probability / bayes_dividers[j];
            }
        }

        // Step5
        // Update mean of each gaussian distribution
        // Update sigma of each gaussian distribution
        // Update probability of each gaussian distribution
        for (lint i = 0; i < this->m_gssns; ++i)
        {
            double sum_x = 0;
            double sum_sigma = 0;
            double sum_p_gaussians = 0;

            for (lint j = 0; j < input_size; ++j)
            {
                double p_g = p_gaussians_in_x[i].m_data[j];
                sum_x += p_g * input.m_data[j];

                double dist = input.m_data[j] - gaussians[i].m_mu;
                sum_sigma += p_g * (dist * dist);

                sum_p_gaussians += p_g;
            }

            double new_mu = sum_x / sum_p_gaussians;
            double new_sigma = sqrt(sum_sigma / sum_p_gaussians);
            double new_probability = sum_p_gaussians / input_size;

            gaussians[i].m_mu = new_mu;
            gaussians[i].m_sigma = new_sigma;
            gaussians[i].m_probability = new_probability;
        }
    }

    delete[]bayes_dividers;

    return gaussians;
}

neurons::EM_1d::~EM_1d()
{}

