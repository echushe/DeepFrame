#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Functions.h"


namespace neurons
{
    class Linear_Regression
    {
    private:
        Matrix m_train_x;
        Matrix m_train_y;
        Matrix m_w;

    public:
        Linear_Regression(const std::vector<Vector> & x, const Vector & y);

        void fit(double accuracy = 10e-14);

        Vector predict(const std::vector<Vector> & x);

        Vector coef_and_intercept() const;

    private:
        Matrix copy_input_into_mat(const std::vector<Vector> & x);
    };
}
