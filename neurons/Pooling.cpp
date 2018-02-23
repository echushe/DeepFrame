#include "Pooling.h"


neurons::Pooling_2d::Pooling_2d(const Shape &input_sh, const Shape &kernel_sh)
    : m_input_sh{ input_sh }, m_kernel_sh{ kernel_sh }
{
    if (kernel_sh.dim() != 4)
    {
        throw std::invalid_argument(
            std::string("neurons::Pooling_2d: Shape of pooling kernel should be compatible with Pooling_2d."));
    }

    if (input_sh[3] != kernel_sh[3])
    {
        throw std::invalid_argument(
            std::string("neurons::Pooling_2d: Numbers of channels should be the same for inputs and filters."));
    }

    this->m_output_sh = Shape{
        input_sh[0],
        input_sh[1] / kernel_sh[1],
        input_sh[2] / kernel_sh[2],
        input_sh[3] };

    this->m_diff_y_to_x = Matrix{ this->m_input_sh };
}

neurons::Matrix neurons::Pooling_2d::operator()(const Matrix & in)
{
    if (in.m_shape != this->m_input_sh)
    {
        throw std::invalid_argument(
            std::string("neurons::Pooling_2d: Shape of input should be compatible with MaxPooling_2d's definition.")
        );
    }

    Matrix output{ this->m_output_sh };
    m_diff_y_to_x = 0;

    lint batch_size = this->m_input_sh[0];

    lint in_rows = this->m_input_sh[1];
    lint in_cols = this->m_input_sh[2];

    lint k_rows = this->m_kernel_sh[1];
    lint k_cols = this->m_kernel_sh[2];

    lint r_stride = k_rows;
    lint c_stride = k_cols;

    lint o_rows = this->m_output_sh[1];
    lint o_cols = this->m_output_sh[2];

    lint chls = this->m_input_sh[3];

    lint in_size = in_rows * in_cols * chls;
    // lint in_r_size = in_cols * chls;
    lint in_r_stride_size = in_cols * chls * r_stride;
    lint in_c_stride_size = chls * c_stride;

    lint o_size = o_rows * o_cols * chls;
    lint o_r_stride_size = o_cols * chls;

    double *in_start = in.m_data;
    double *diff_start = this->m_diff_y_to_x.m_data;
    double *o_start = output.m_data;

    for (lint i = 0; i < batch_size; ++i)
    {
        double *in_r_stride_start = in_start;
        double *diff_r_stride_start = diff_start;
        double *o_r_stride_start = o_start;

        for (lint o_r = 0; o_r < o_rows; ++o_r)
        {
            double *in_c_stride_start = in_r_stride_start;
            double *diff_c_stride_start = diff_r_stride_start;
            double *o_c_stride_start = o_r_stride_start;

            for (lint o_c = 0; o_c < o_cols; ++o_c)
            {
                this->pooling_func(in_c_stride_start, diff_c_stride_start, o_c_stride_start, in_cols, chls, k_cols, k_rows);

                in_c_stride_start += in_c_stride_size;
                diff_c_stride_start += in_c_stride_size;
                o_c_stride_start += chls;
            }

            in_r_stride_start += in_r_stride_size;
            diff_r_stride_start += in_r_stride_size;
            o_r_stride_start += o_r_stride_size;
        }

        in_start += in_size;
        diff_start += in_size;
        o_start += o_size;
    }

    // std::cout << diff_x << '\n';

    return output;
}

neurons::Matrix neurons::Pooling_2d::back_propagate(const Matrix & diff_E_to_output) const
{
    if (diff_E_to_output.m_shape != this->m_output_sh)
    {
        throw std::invalid_argument(
            std::string("neurons::Pooling_2d::back_propagate: Shape of derivative should be compatible with MaxPooling_2d's definition.")
        );
    }

    // diff_E_to_input should be initialized with zero
    Matrix diff_E_to_input{ this->m_input_sh, 0 };

    lint batch_size = this->m_input_sh[0];

    lint in_rows = this->m_input_sh[1];
    lint in_cols = this->m_input_sh[2];

    lint k_rows = this->m_kernel_sh[1];
    lint k_cols = this->m_kernel_sh[2];

    lint r_stride = k_rows;
    lint c_stride = k_cols;

    lint o_rows = this->m_output_sh[1];
    lint o_cols = this->m_output_sh[2];

    lint chls = this->m_input_sh[3];

    lint in_size = in_rows * in_cols * chls;
    // lint in_r_size = in_cols * chls;
    lint in_r_stride_size = in_cols * chls * r_stride;
    lint in_c_stride_size = chls * c_stride;

    lint o_size = o_rows * o_cols * chls;
    lint o_r_stride_size = o_cols * chls;

    double *diff_E_to_x_start = diff_E_to_input.m_data;
    double *diff_x_start = this->m_diff_y_to_x.m_data;
    double *diff_E_to_y_start = diff_E_to_output.m_data;

    for (lint i = 0; i < batch_size; ++i)
    {
        double *diff_E_to_x_r_stride_start = diff_E_to_x_start;
        double *diff_x_r_stride_start = diff_x_start;
        double *diff_E_to_y_r_stride_start = diff_E_to_y_start;

        for (lint o_r = 0; o_r < o_rows; ++o_r)
        {
            double *diff_E_to_x_c_stride_start = diff_E_to_x_r_stride_start;
            double *diff_x_c_stride_start = diff_x_r_stride_start;
            double *diff_E_to_y_c_stride_start = diff_E_to_y_r_stride_start;

            for (lint o_c = 0; o_c < o_cols; ++o_c)
            {
                this->back_propagate_func(
                    diff_x_c_stride_start,
                    diff_E_to_y_c_stride_start,
                    diff_E_to_x_c_stride_start,
                    in_cols, chls, k_cols, k_rows);

                diff_E_to_x_c_stride_start += in_c_stride_size;
                diff_x_c_stride_start += in_c_stride_size;
                diff_E_to_y_c_stride_start += chls;
            }

            diff_E_to_x_r_stride_start += in_r_stride_size;
            diff_x_r_stride_start += in_r_stride_size;
            diff_E_to_y_r_stride_start += o_r_stride_size;
        }

        diff_E_to_x_start += in_size;
        diff_x_start += in_size;
        diff_E_to_y_start += o_size;
    }

    return diff_E_to_input;
}

neurons::Shape neurons::Pooling_2d::get_output_shape() const
{
    return this->m_output_sh;
}

neurons::MaxPooling_2d::MaxPooling_2d(const Shape & input_sh, const Shape & kernel_sh)
    : Pooling_2d(input_sh, kernel_sh)
{}

std::unique_ptr<neurons::Pooling_2d> neurons::MaxPooling_2d::clone()
{
    return std::make_unique<MaxPooling_2d>(*this);
}

void neurons::MaxPooling_2d::pooling_func(const double *input_data, double *diff_data, double *output_data, lint in_cols, lint chls, lint k_cols, lint k_rows)
{
    const double *k_ch_start = input_data;
    double *k_d_ch_start = diff_data;

    double lowest = std::numeric_limits<double>::max() * (-1);

    lint in_r_size = in_cols * chls;

    for (lint ch = 0; ch < chls; ++ch)
    {
        double max = lowest;
        lint argmax_r = 0, argmax_c = 0;

        const double *k_r_start = k_ch_start;

        for (lint k_r = 0; k_r < k_rows; ++k_r)
        {
            const double *k_c_start = k_r_start;

            for (lint k_c = 0; k_c < k_cols; ++k_c)
            {
                // std::cout << *k_c_start << "  ";
                if (*k_c_start > max)
                {
                    max = *k_c_start;
                    argmax_r = k_r;
                    argmax_c = k_c;
                }
                k_c_start += chls;
            }

            k_r_start += in_r_size;
        }

        // std::cout << max << '\n';
        output_data[ch] = max;
        *(k_d_ch_start + argmax_r * in_r_size + argmax_c * chls) = 1;

        ++k_ch_start;
        ++k_d_ch_start;
    }
}

void neurons::MaxPooling_2d::back_propagate_func(
    const double * diff_y_to_x_data,
    const double * diff_E_to_y_data,
    double * diff_E_to_x_data,
    lint in_cols, lint chls, lint k_cols, lint k_rows) const
{
    const double *d_y_to_x_ch_start = diff_y_to_x_data;
    double *d_E_to_x_ch_start = diff_E_to_x_data;

    lint in_r_size = in_cols * chls;

    for (lint ch = 0; ch < chls; ++ch)
    {
        const double *k_r_y_to_x_start = d_y_to_x_ch_start;
        double *k_r_E_to_x_start = d_E_to_x_ch_start;

        for (lint k_r = 0; k_r < k_rows; ++k_r)
        {
            const double *k_c_y_to_x_start = k_r_y_to_x_start;
            double *k_c_E_to_x_start = k_r_E_to_x_start;

            for (lint k_c = 0; k_c < k_cols; ++k_c)
            {
                /*
                if (0 != *k_c_y_to_x_start)
                {
                    *k_c_E_to_x_start = diff_E_to_y_data[ch];
                }
                else
                {
                    *k_c_E_to_x_start = 0;
                }
                */

                *k_c_E_to_x_start = *k_c_y_to_x_start * diff_E_to_y_data[ch];
                
                k_c_y_to_x_start += chls;
                k_c_E_to_x_start += chls;
            }

            k_r_y_to_x_start += in_r_size;
            k_r_E_to_x_start += in_r_size;
        }

        ++d_y_to_x_ch_start;
        ++d_E_to_x_ch_start;
    }
}



neurons::Pooling_layer::Pooling_layer()
{}

neurons::Pooling_layer::Pooling_layer(const Shape & input_sh, const Shape & kernel_sh, lint threads)
    : m_input_sh{ input_sh }, m_kernel_sh{ kernel_sh }, m_ops{ static_cast<size_t>(threads) }
{
    for (lint i = 0; i < threads; ++i)
    {
        this->m_ops[i] = std::make_shared<Pooling_layer_op>(input_sh, kernel_sh);
    }
}

std::vector<std::shared_ptr<neurons::Pooling_layer_op>>& neurons::Pooling_layer::operation_instances() const
{
    return this->m_ops;
}

neurons::Shape neurons::Pooling_layer::output_shape() const
{
    if (this->m_ops.empty())
    {
        return MaxPooling_2d{ this->m_input_sh, this->m_kernel_sh }.get_output_shape();
    }
    else
    {
        return this->m_ops[0]->output_shape();
    }
}


neurons::Pooling_layer_op::Pooling_layer_op()
{}

neurons::Pooling_layer_op::Pooling_layer_op(const Shape & input_sh, const Shape & kernel_sh)
    : m_input_sh{ input_sh }, m_kernel_sh{ kernel_sh }
{}

std::vector<neurons::Matrix> neurons::Pooling_layer_op::forward_propagate(const std::vector<Matrix>& inputs)
{
    size_t samples = inputs.size();
    // this->m_pools.resize(samples);
    this->m_pools.clear();

    std::vector<Matrix> output{ samples };

    for (size_t i = 0; i < samples; ++i)
    {
        this->m_pools.push_back(MaxPooling_2d{ this->m_input_sh, this->m_kernel_sh });
        
        // Get output of pooling and feed it to nn layer
        output[i] = this->m_pools[i](inputs[i]);
        // std::cout << inputs[i];
    }

    return output;
}

std::vector<neurons::Matrix> neurons::Pooling_layer_op::back_propagate(const std::vector<Matrix>& E_to_y_diffs)
{
    size_t samples = this->m_pools.size();
    std::vector<Matrix> E_to_x_diffs{ samples };

    for (size_t i = 0; i < samples; ++i)
    {
        E_to_x_diffs[i] = this->m_pools[i].back_propagate(E_to_y_diffs[i]);
    }

    return E_to_x_diffs;
}

neurons::Shape neurons::Pooling_layer_op::output_shape() const
{
    if (this->m_pools.empty())
    {
        return MaxPooling_2d{ this->m_input_sh, this->m_kernel_sh }.get_output_shape();
    }
    else
    {
        return this->m_pools[0].get_output_shape();
    }
}

