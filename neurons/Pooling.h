#pragma once
#include "Matrix.h"

namespace neurons
{
    class Pooling_2d
    {
    protected:
        Shape m_input_sh;
        Shape m_kernel_sh;
        Shape m_output_sh;

        mutable Matrix m_diff_y_to_x;

    public:
        Pooling_2d() {}
        Pooling_2d(const Shape &input_sh, const Shape &kernel_sh);

        Shape get_output_shape() const;

        virtual std::unique_ptr<Pooling_2d> clone() = 0;

        Matrix operator () (const Matrix & in);

        Matrix back_propagate(const Matrix & diff_E_to_output) const;

    private:
        virtual void pooling_func(const double *input_data, double *diff_data,
            double *output_data, lint in_cols, lint chls, lint k_cols, lint k_rows) = 0;

        virtual void back_propagate_func(const double *diff_y_to_x_data, const double *diff_E_to_y_data,
            double *diff_E_to_x_data, lint in_cols, lint chls, lint k_cols, lint k_rows) const = 0;
    };


    class MaxPooling_2d : public Pooling_2d
    {
    public:
        MaxPooling_2d() {}
        MaxPooling_2d(const Shape &input_sh, const Shape &kernel_sh);

        virtual std::unique_ptr<Pooling_2d> clone();

    private:
        virtual void pooling_func(const double *input_data, double *diff_data,
            double *output_data, lint in_cols, lint chls, lint k_cols, lint k_rows);

        virtual void back_propagate_func(const double *diff_y_to_x_data, const double *diff_E_to_y_data,
            double *diff_E_to_x_data, lint in_cols, lint chls, lint k_cols, lint k_rows) const;
    };

    class Pooling_layer_op;
    class Pooling_layer
    {
    private:
        Shape m_input_sh;
        Shape m_kernel_sh;

        mutable std::vector<std::shared_ptr<Pooling_layer_op>> m_ops;
    
    public:
        Pooling_layer();
        Pooling_layer(const Shape &input_sh, const Shape &kernel_sh, lint threads);

        std::vector<std::shared_ptr<Pooling_layer_op>>& operation_instances() const;

        Shape output_shape() const;
    };

    class Pooling_layer_op
    {
    private:
        Shape m_input_sh;
        Shape m_kernel_sh;

        std::vector<MaxPooling_2d> m_pools;

    public:
        Pooling_layer_op();
        Pooling_layer_op(const Shape &input_sh, const Shape &kernel_sh);

        std::vector<Matrix> forward_propagate(const std::vector<Matrix> &inputs);

        virtual std::vector<Matrix> back_propagate(const std::vector<Matrix> & E_to_y_diffs);

        Shape output_shape() const;
    };
}
