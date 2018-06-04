#include "Convolution.h"

neurons::Conv_1d::Conv_1d(const Shape & input_shape, const Shape & weights_shape, lint stride)
    : m_input_sh{ input_shape }, m_weights_sh{ weights_shape }, m_stride{ stride }
{
    if (input_shape.dim() < 3 || weights_shape.dim() < 3)
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_1d: Shape of inputs and weights should be compatible with the this convolution."));
    }

    lint out_rows = (this->m_input_sh[1] - this->m_weights_sh[0]) / this->m_stride + 1;

    this->m_diff_to_w = TMatrix<>{ Shape{
        this->m_input_sh[0],
        this->m_weights_sh[0],
        this->m_weights_sh[1],
        out_rows
    }, 0 };
}

neurons::Conv_1d::Conv_1d()
{
}

neurons::Conv_1d::~Conv_1d()
{}

neurons::TMatrix<> neurons::Conv_1d::operator()(const TMatrix<> & input, const TMatrix<> & weights, const TMatrix<> & bias)
{
    if (this->m_input_sh.dim() < 3 || this->m_weights_sh.dim() < 3 || bias.m_shape.dim() < 2)
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_1d: Shape of inputs and weights should be compatible with the this convolution."));
    }
    
    lint in_dim = this->m_input_sh.dim();
    lint w_dim = this->m_weights_sh.dim();

    Shape in_batch_sh = this->m_input_sh.sub_shape(0, in_dim - 3);
    Shape in_conv_sh = this->m_input_sh.sub_shape(in_dim - 2, in_dim - 1);

    Shape w_conv_sh = this->m_weights_sh.sub_shape(0, 1);
    Shape w_batch_sh = this->m_weights_sh.sub_shape(2, w_dim - 1);

    lint in_rows = in_conv_sh[0];
    lint w_rows = this->m_weights_sh[0];

    if (in_rows < w_rows)
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_1d: size of input should be no less than size of the filter."));
    }

    lint in_batch_size = in_batch_sh.size();
    lint in_conv_size = in_conv_sh.size();

    // lint w_conv_size = w_conv_sh.size();
    lint w_batch_size = w_batch_sh.size();

    lint chls = this->m_input_sh[in_dim - 1];
    lint out_rows = (in_rows - w_rows) + 1;
    
    TMatrix<> out{ in_batch_sh + Shape{ out_rows } + w_batch_sh };
    lint out_size = out_rows * w_batch_size;

    double *in_start = input.m_data;
    double *out_start = out.m_data;

    for (lint i = 0; i < in_batch_size; ++i)
    {
        double *w_start = weights.m_data;
        double *b_p = bias.m_data;
        double *out_l_start = out_start;
        // Go through each filter
        for (lint m = 0; m < w_batch_size; ++m)
        {
            double *stride_start = in_start;
            double *out_row_start = out_l_start;
            // Next stride
            for (lint stride_pos = 0; stride_pos < out_rows; ++stride_pos)
            {
                double *in_row_start = stride_start;
                double *w_row_start = w_start;

                double sum = 0;

                // Go through each pixel and each channel
                for (lint k = 0; k < w_rows; ++k)
                {
                    double *in_p = in_row_start;
                    double *w_p = w_row_start;

                    for (lint j = 0; j < chls; ++j)
                    {
                        //std::cout << *in_p << '\t';
                        sum += *in_p * *w_p;

                        ++in_p;
                        w_p += w_batch_size;
                    }

                    in_row_start += chls;
                    w_row_start += chls * w_batch_size;
                    //std::cout << '\n';
                }

                *out_row_start = sum + *b_p;
                out_row_start += w_batch_size;
                stride_start += chls * this->m_stride;
                //std::cout << '\n';
            }
            
            ++w_start;
            ++b_p;
            ++out_l_start;
        }

        in_start += in_conv_size;
        out_start += out_size;
    }


    // Calculate the Derivative [ d(conv_product) / d(x) ]
    double *diff_start = this->m_diff_to_w.m_data;
    lint diff_size = w_rows * chls * out_rows;
    lint chls_out_rows = chls * out_rows;
    in_start = input.m_data;

    for (lint i = 0; i < in_batch_size; ++i)
    {
        double *stride_row_start = in_start;
        double *diff_stride_start = diff_start;
        // Next stride (row)
        for (lint stride_r_pos = 0; stride_r_pos < out_rows; ++stride_r_pos)
        {
            double *in_row_start = stride_row_start;
            double *diff_w_r_start = diff_stride_start;

            // Go through each pixel and each channel
            for (lint r = 0; r < w_rows; ++r)
            {
                double *in_p = in_row_start;
                double *diff_chls_start = diff_w_r_start;

                for (lint j = 0; j < chls; ++j)
                {
                    *diff_chls_start = *in_p;

                    ++in_p;
                    diff_chls_start += out_rows;
                }

                in_row_start += chls;
                diff_w_r_start += chls_out_rows;
            }

            stride_row_start += chls * this->m_stride;
            ++diff_stride_start;
        }

        in_start += in_conv_size;
        diff_start += diff_size;
    }

    return out;
}

neurons::TMatrix<> & neurons::Conv_1d::get_diff_to_weights() const
{
    return this->m_diff_to_w;
}


neurons::Conv_2d::Conv_2d(
    const Shape & input_shape,
    const Shape & weights_shape,
    lint r_stride, lint c_stride,
    lint r_zero_p, lint c_zero_p)
    :
    m_input_sh{ input_shape }, m_weights_sh{ weights_shape },
    m_r_stride{ r_stride }, m_c_stride{ c_stride },
    m_r_zero_p{ r_zero_p }, m_c_zero_p{ c_zero_p }
{
    if (input_shape.dim() != 4 || weights_shape.dim() != 4)
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_2d: Shape of inputs and weights should be compatible with the this convolution."));
    }

    lint in_batch_size = this->m_input_sh[0];
    lint in_rows = this->m_input_sh[1];
    lint in_cols = this->m_input_sh[2];
    lint chls = this->m_input_sh[3];

    if (chls != this->m_weights_sh[2])
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_2d: Numbers of channels should be the same for inputs and filters."));
    }

    in_rows += 2 * m_r_zero_p;
    in_cols += 2 * m_c_zero_p;

    this->m_ex_in_sh = Shape{ in_batch_size, in_rows, in_cols } +
        this->m_input_sh.sub_shape(3, this->m_input_sh.dim() - 1);

    lint out_rows = (in_rows - this->m_weights_sh[0]) / this->m_r_stride + 1;
    lint out_cols = (in_cols - this->m_weights_sh[1]) / this->m_c_stride + 1;

    this->m_output_sh = Shape{ in_batch_size, out_rows, out_cols, this->m_weights_sh[this->m_weights_sh.dim() - 1] };

    this->m_diff_to_w = TMatrix<>{ Shape{
        this->m_input_sh[0],
        this->m_weights_sh[0],
        this->m_weights_sh[1],
        this->m_weights_sh[2],
        out_rows,
        out_cols,
    }, 0 };

    this->m_diff_to_x = TMatrix<>{ Shape{
        this->m_input_sh[0],
        this->m_input_sh[1],
        this->m_input_sh[2],
        this->m_input_sh[3],
        out_rows,
        out_cols,
        this->m_weights_sh[3] }, 0 };
}

neurons::Conv_2d::Conv_2d()
{
}

neurons::Conv_2d::~Conv_2d()
{}

neurons::TMatrix<> neurons::Conv_2d::operator()(const TMatrix<> & input, const TMatrix<> & weights, const TMatrix<> & bias)
{
    if (this->m_input_sh != input.shape() || this->m_weights_sh != weights.shape() || bias.m_shape.dim() < 2)
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_1d: Shape of inputs and weights should be compatible with the this convolution."));
    }

    TMatrix<> ex_input = zero_padding(input);

    lint in_dim = this->m_ex_in_sh.dim();
    lint w_dim = this->m_ex_in_sh.dim();

    Shape in_batch_sh = this->m_ex_in_sh.sub_shape(0, in_dim - 4);
    Shape in_conv_sh = this->m_ex_in_sh.sub_shape(in_dim - 3, in_dim - 1);

    Shape w_conv_sh = this->m_weights_sh.sub_shape(0, 2);
    Shape w_batch_sh = this->m_weights_sh.sub_shape(3, w_dim - 1);

    lint in_rows = in_conv_sh[0];
    lint in_cols = in_conv_sh[1];
    lint w_rows = w_conv_sh[0];
    lint w_cols = w_conv_sh[1];

    // lint in_size = in_rows * in_cols;
    // lint w_size = w_rows * w_cols;

    if (in_rows < w_rows || in_cols < w_cols)
    {
        throw std::invalid_argument(
            std::string("neurons::Conv_1d: size of input should be no less than size of the filter."));
    }

    lint in_batch_size = in_batch_sh.size();
    lint in_conv_size = in_conv_sh.size();

    // lint w_conv_size = w_conv_sh.size();
    lint w_batch_size = w_batch_sh.size();

    lint chls = this->m_ex_in_sh[in_dim - 1];
    lint out_rows = (in_rows - w_rows) / this->m_r_stride + 1;
    lint out_cols = (in_cols - w_cols) / this->m_c_stride + 1;

    TMatrix<> out{ in_batch_sh + Shape{ out_rows, out_cols } + w_batch_sh };
    lint out_size = out_rows * out_cols * w_batch_size;

    lint in_cols_chls = in_cols * chls;
    lint w_cols_chls_w_batch_size = w_cols * chls * w_batch_size;
    lint chls_w_batch_size = chls * w_batch_size;

    lint chls_c_stride = chls * this->m_c_stride;
    lint in_cols_chls_r_stride = in_cols_chls * this->m_r_stride;
    lint out_cols_w_batch_size = out_cols * w_batch_size;

    double *in_start = ex_input.m_data;
    double *out_start = out.m_data;

    double *w_start;
    double *b_p;
    double *out_l_start;

    double *stride_row_start;
    double *out_row_start;

    double *stride_col_start;
    double *out_col_start;

    double *in_row_start;
    double *w_row_start;
    double sum = 0;

    double *in_col_start;
    double *w_col_start;

    double *in_p;
    double *w_p;

    for (lint i = 0; i < in_batch_size; ++i)
    {
        w_start = weights.m_data;
        b_p = bias.m_data;
        out_l_start = out_start;

        // Go through each filter
        for (lint m = 0; m < w_batch_size; ++m)
        {
            stride_row_start = in_start;
            out_row_start = out_l_start;
            // Next stride
            for (lint stride_r_pos = 0; stride_r_pos < out_rows; ++stride_r_pos)
            {
                stride_col_start = stride_row_start;
                out_col_start = out_row_start;

                for (lint stride_c_pos = 0; stride_c_pos < out_cols; ++stride_c_pos)
                {
                    in_row_start = stride_col_start;
                    w_row_start = w_start;
                    sum = 0;

                    // Go through each pixel and each channel
                    for (lint r = 0; r < w_rows; ++r)
                    {
                        in_col_start = in_row_start;
                        w_col_start = w_row_start;

                        for (lint c = 0; c < w_cols; ++c)
                        {
                            in_p = in_col_start;
                            w_p = w_col_start;

                            for (lint j = 0; j < chls; ++j)
                            {
                                //std::cout << *in_p << '\t';
                                sum += *in_p * *w_p;

                                ++in_p;
                                w_p += w_batch_size;
                            }

                            in_col_start += chls;
                            w_col_start += chls_w_batch_size;
                        }

                        in_row_start += in_cols_chls;
                        w_row_start += w_cols_chls_w_batch_size;
                        //std::cout << '\n';
                    }

                    *out_col_start = sum + *b_p;
                    out_col_start += w_batch_size;
                    stride_col_start += chls_c_stride;
                }

                out_row_start += out_cols_w_batch_size;
                stride_row_start += in_cols_chls_r_stride;
                //std::cout << '\n';
            }

            ++w_start;
            ++b_p;
            ++out_l_start;
        }

        in_start += in_conv_size;
        out_start += out_size;
    }


    {
        // Calculate the Derivative [ d(conv_product) / d(w) ]
        double *diff_start = this->m_diff_to_w.m_data;
        lint diff_size = w_rows * w_cols * chls * out_rows * out_cols;
        lint w_cols_chls_out_rows_out_cols = w_cols * chls * out_rows * out_cols;
        lint chls_out_rows_out_cols = chls * out_rows * out_cols;
        lint out_rows_out_cols = out_rows * out_cols;

        in_start = ex_input.m_data;

        double *stride_row_start;
        double *diff_stride_r_start;

        double *stride_col_start;
        double *diff_stride_c_start;

        double *in_row_start;
        double *diff_w_r_start;

        double *in_col_start;
        double *diff_w_c_start;

        double *in_p;
        double *diff_chls_start;

        for (lint i = 0; i < in_batch_size; ++i)
        {
            stride_row_start = in_start;
            diff_stride_r_start = diff_start;
            // Next stride (row)
            for (lint stride_r_pos = 0; stride_r_pos < out_rows; ++stride_r_pos)
            {
                stride_col_start = stride_row_start;
                diff_stride_c_start = diff_stride_r_start;

                // Next stride (column)
                for (lint stride_c_pos = 0; stride_c_pos < out_cols; ++stride_c_pos)
                {
                    in_row_start = stride_col_start;
                    diff_w_r_start = diff_stride_c_start;

                    // Go through each pixel and each channel
                    for (lint r = 0; r < w_rows; ++r)
                    {
                        in_col_start = in_row_start;
                        diff_w_c_start = diff_w_r_start;

                        for (lint c = 0; c < w_cols; ++c)
                        {
                            in_p = in_col_start;
                            diff_chls_start = diff_w_c_start;

                            for (lint j = 0; j < chls; ++j)
                            {
                                *diff_chls_start = *in_p;

                                ++in_p;
                                diff_chls_start += out_rows_out_cols;
                            }

                            in_col_start += chls;
                            diff_w_c_start += chls_out_rows_out_cols;
                        }

                        in_row_start += in_cols_chls;
                        diff_w_r_start += w_cols_chls_out_rows_out_cols;
                    }

                    stride_col_start += chls * this->m_c_stride;
                    ++diff_stride_c_start;
                }

                stride_row_start += in_cols_chls * this->m_r_stride;
                diff_stride_r_start += out_cols;
            }

            in_start += in_conv_size;
            diff_start += diff_size;
        }
    }


    {
        // Calculate the Derivative [ d(conv_product) / d(x) ]
        double *diff_start = this->m_diff_to_x.m_data;
        lint _in_rows = in_rows - 2 * this->m_r_zero_p;
        lint _in_cols = in_cols - 2 * this->m_c_zero_p;

        lint diff_size = _in_rows * _in_cols * chls * out_rows * out_cols * w_batch_size;
        lint diff_r_size = _in_cols * chls * out_rows * out_cols * w_batch_size;
        lint diff_c_size = chls * out_rows * out_cols * w_batch_size;
        lint diff_ch_size = out_rows * out_cols * w_batch_size;
        lint diff_o_r_size = out_cols * w_batch_size;
        lint diff_o_c_size = w_batch_size;

        lint in_rows_minus_padding = in_rows - this->m_r_zero_p;
        lint in_cols_minus_padding = in_cols - this->m_c_zero_p;

        double *w_start = weights.m_data;

        double *diff_r_start;
        double *w_in_r_start;

        double *diff_c_start;
        double *w_in_c_start;

        double *diff_ch_start;
        double *w_chls_start;

        double *diff_o_r_start;
        double *w_r_start;

        double *diff_o_c_start;
        double *w_c_start;

        for (lint i = 0; i < in_batch_size; ++i)
        {
            diff_r_start = diff_start;
            w_in_r_start = w_start;

            for (lint in_r = 0; in_r < in_rows; ++in_r)
            {
                w_in_c_start = w_in_r_start;

                if (in_r >= this->m_r_zero_p && in_r < in_rows_minus_padding)
                {
                    diff_c_start = diff_r_start;

                    for (lint in_c = 0; in_c < in_cols; ++in_c)
                    {
                        w_chls_start = w_in_c_start;

                        if (in_c >= this->m_c_zero_p && in_c < in_cols_minus_padding)
                        {
                            diff_ch_start = diff_c_start;

                            for (lint in_chls = 0; in_chls < chls; ++in_chls)
                            {
                                diff_o_r_start = diff_ch_start;
                                w_r_start = w_chls_start;

                                for (lint out_r = 0; out_r < out_rows; ++out_r)
                                {
                                    lint w_row = in_r - out_r;

                                    if (w_row < 0 || w_row >= w_rows)
                                    {
                                        for (lint i = 0; i < diff_o_r_size; ++i)
                                        {
                                            diff_o_r_start[i] = 0;
                                        }
                                    }
                                    else
                                    {
                                        diff_o_c_start = diff_o_r_start;
                                        w_c_start = w_r_start;

                                        for (lint out_c = 0; out_c < out_cols; ++out_c)
                                        {
                                            lint w_col = in_c - out_c;

                                            if (w_col < 0 || w_col >= w_cols)
                                            {
                                                for (lint w = 0; w < w_batch_size; ++w)
                                                {
                                                    diff_o_c_start[w] = 0;
                                                }
                                            }
                                            else
                                            {
                                                for (lint w = 0; w < w_batch_size; ++w)
                                                {
                                                    diff_o_c_start[w] = w_c_start[w];
                                                }
                                            }

                                            diff_o_c_start += diff_o_c_size;
                                            w_c_start -= chls_w_batch_size;
                                        }
                                    }

                                    diff_o_r_start += diff_o_r_size;
                                    w_r_start -= w_cols_chls_w_batch_size;
                                }

                                diff_ch_start += diff_ch_size;
                                w_chls_start += w_batch_size;
                            }

                            diff_c_start += diff_c_size;
                        }

                        w_in_c_start += chls_w_batch_size;
                    }

                    diff_r_start += diff_r_size;
                }

                w_in_r_start += w_cols_chls_w_batch_size;
            }

            diff_start += diff_size;
        }
    }

    return out;
}


neurons::TMatrix<> & neurons::Conv_2d::get_diff_to_weights() const
{
    return this->m_diff_to_w;
}


neurons::TMatrix<> & neurons::Conv_2d::get_diff_to_input() const
{
    return this->m_diff_to_x;
}


neurons::Shape neurons::Conv_2d::get_output_shape() const
{
    return this->m_output_sh;
}


neurons::TMatrix<> neurons::Conv_2d::zero_padding(const TMatrix<> & input)
{
    if (this->m_r_zero_p < 0)
    {
        this->m_r_zero_p = 0;
    }

    if (this->m_c_zero_p < 0)
    {
        this->m_c_zero_p = 0;
    }

    lint in_batch_size = this->m_input_sh[0];
    lint in_rows = this->m_input_sh[1];
    lint in_cols = this->m_input_sh[2];
    lint chls = this->m_input_sh[3];

    lint ex_in_rows = in_rows + 2 * m_r_zero_p;
    lint ex_in_cols = in_cols + 2 * m_c_zero_p;

    TMatrix<> ex_input{
        Shape{ in_batch_size, ex_in_rows, ex_in_cols } +
        this->m_input_sh.sub_shape(3, this->m_input_sh.dim() - 1) };

    double *in_start = input.m_data;
    double *ex_in_start = ex_input.m_data;

    lint in_size = in_rows * in_cols * chls;
    lint ex_in_size = ex_in_rows * ex_in_cols * chls;

    lint in_r_size = in_cols * chls;
    lint ex_in_r_size = ex_in_cols * chls;

    for (lint i = 0; i < in_batch_size; ++i)
    {
        double *in_r_start = in_start;
        double *ex_in_r_start = ex_in_start;
        
        for (lint j = 0; j < ex_in_rows; ++j)
        {
            if (j >= m_r_zero_p && j < ex_in_rows - m_r_zero_p)
            {
                double *in_c_start = in_r_start;
                double *ex_in_c_start = ex_in_r_start;

                for (lint k = 0; k < ex_in_cols; ++k)
                {
                    if (k >= m_c_zero_p && k < ex_in_cols - m_c_zero_p)
                    {
                        for (lint l = 0; l < chls; ++l)
                        {
                            ex_in_c_start[l] = in_c_start[l];
                        }

                        in_c_start += chls;
                    }
                    else
                    {
                        for (lint l = 0; l < chls; ++l)
                        {
                            ex_in_c_start[l] = 0;
                        }
                    }

                    ex_in_c_start += chls;
                }

                in_r_start += in_r_size;
            }
            else
            {
                for (lint k = 0; k < ex_in_r_size; ++k)
                {
                    ex_in_r_start[k] = 0;
                }
            }

            ex_in_r_start += ex_in_r_size;
        }

        in_start += in_size;
        ex_in_start += ex_in_size;
    }

    // std::cout << ex_input << '\n';

    return ex_input;
}
