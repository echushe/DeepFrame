#include "LinearRegression.h"


neurons::Linear_Regression::Linear_Regression(const std::vector<Vector> & x, const Vector & y)
    :
    m_train_y{ y },
    m_w{ Shape{ 1, x[0].dim() + 1 }}
{
    this->m_train_y = transpose(this->m_train_y);

    // Copy values of x to training x
    this->m_train_x = this->copy_input_into_mat(x);

    // Initialize weights and bias
    double var = static_cast<double>(10) / this->m_w.shape()[1];
    m_w.gaussian_random(0, var);
}


void neurons::Linear_Regression::fit(double accuracy)
{
    lint step = 0;
    double l_rate = static_cast<double>(2) / this->m_train_x.shape().size();
    if (l_rate > 0.01)
    {
        l_rate = 0.01;
    }
    
    while (true)
    {
        // Linear multiplication which resembles Forward propagation
        // of neural network
        TMatrix<> y = transpose(this->m_train_x * transpose(this->m_w));

        // Calculate dE/dy
        TMatrix<> diff_E_to_y = 2.0 * (y - this->m_train_y);
        
        // Calculate dE/dw = (dE/dy) * (dy/dw)
        TMatrix<> gradient = diff_E_to_y * this->m_train_x;

        if (gradient.euclidean_norm() < accuracy)
        {
            std::cout << "step: " << step << " norm: " << gradient.euclidean_norm() << "\n";
            break;
        }

        // Update the weights that resembles back propagation of
        // neural network
        this->m_w -= gradient * l_rate;
        ++step;
    }
}


neurons::Vector neurons::Linear_Regression::predict(const std::vector<Vector>& x)
{
    TMatrix<> test_x = this->copy_input_into_mat(x);
    TMatrix<> y = transpose(test_x * transpose(this->m_w));

    return y.flaten();
}

neurons::Vector neurons::Linear_Regression::coef_and_intercept() const
{
    return this->m_w.flaten();
}

neurons::TMatrix<> neurons::Linear_Regression::copy_input_into_mat(const std::vector<Vector>& x)
{
    TMatrix<> ret{ Shape{ static_cast<lint>(x.size()), x[0].dim() + 1 } };
    // Copy values of x to test x
    double * test_x_start = ret.m_data;
    lint columns = x[0].dim() + 1;
    lint x_dims = columns - 1;

    for (auto itr = x.begin(); itr != x.end(); ++itr)
    {
        memcpy(test_x_start, itr->m_data, x_dims * sizeof(double));
        test_x_start[x_dims] = 1;
        test_x_start += columns;
    }

    return ret;
}






