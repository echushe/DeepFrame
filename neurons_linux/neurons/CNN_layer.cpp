#include "CNN_layer.h"


std::string neurons::CNN_layer::from_binary_data(
    char * binary_data, lint & data_size, TMatrix<>& w, TMatrix<>& b, lint & stride, lint & padding,
    std::unique_ptr<Activation>& act_func, std::unique_ptr<ErrorFunction>& err_func, char *& residual_data, lint & residual_len)
{
    char * position;
    lint re_len;
    std::string nn_type = neurons::Traditional_NN_layer::from_binary_data(binary_data, data_size, w, b, act_func, err_func, position, re_len);

    if ("FCNN" == nn_type)
    {
        residual_data = position;
        residual_len = re_len;
    }
    else
    {
        stride = *(reinterpret_cast<lint *>(position));
        position += sizeof(lint);
        padding = *(reinterpret_cast<lint *>(position));

        residual_len = re_len - 2 * sizeof(lint);
        residual_data = position;
    }

    return nn_type;

}

neurons::CNN_layer::CNN_layer()
{}

neurons::CNN_layer::CNN_layer(
    double mmt_rate,
    lint rows,
    lint cols,
    lint chls,
    lint filters,
    lint filter_rows,
    lint filter_cols,
    lint stride,
    lint padding,
    lint threads,
    neurons::Activation *act_func,
    neurons::ErrorFunction *err_func)
    :
    Traditional_NN_layer(mmt_rate, neurons::Shape{ filter_rows, filter_cols, chls, filters }, neurons::Shape{ 1, filters }, threads, act_func, err_func ),
    m_conv2d{ neurons::Shape{ 1, rows, cols, chls }, neurons::Shape{ filter_rows, filter_cols, chls, filters }, stride, stride, padding, padding }
{
    double var = static_cast<double>(100) / this->m_w.shape().size();
    this->m_w.gaussian_random(0, var);
    this->m_b.gaussian_random(0, var);

    for (lint i = 0; i < threads; ++i)
    {
        this->m_ops[i] = std::make_shared<CNN_layer_op>(this->m_conv2d, this->m_w, this->m_b, this->m_act_func, this->m_err_func);
    }
}

neurons::CNN_layer::CNN_layer(
    double mmt_rate, lint rows, lint cols, lint chls, lint stride, lint padding, lint threads,
    const TMatrix<>& w, const TMatrix<>& b, 
    std::unique_ptr<Activation>& act_func, std::unique_ptr<ErrorFunction>& err_func)
    :
    Traditional_NN_layer(mmt_rate, threads, w, b, act_func, err_func),
    m_conv2d{
        neurons::Shape{ 1, rows, cols, chls },
        neurons::Shape{ w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3] }, stride, stride, padding, padding }
{
    for (lint i = 0; i < threads; ++i)
    {
        this->m_ops[i] = std::make_shared<CNN_layer_op>(this->m_conv2d, this->m_w, this->m_b, this->m_act_func, this->m_err_func);
    }
}


neurons::CNN_layer::CNN_layer(const CNN_layer & other)
    : 
    Traditional_NN_layer(other),
    m_conv2d{ other.m_conv2d }
{
    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::make_shared<CNN_layer_op>(
            *(dynamic_cast<CNN_layer_op*>(other.m_ops[i].get())));
    }
}


neurons::CNN_layer::CNN_layer(CNN_layer && other)
    : Traditional_NN_layer(other),
    m_conv2d{ std::move(other.m_conv2d) }
{
    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::move(other.m_ops[i]);
    }
}


neurons::CNN_layer & neurons::CNN_layer::operator = (const CNN_layer & other)
{
    NN_layer::operator = (other);
    this->m_conv2d = other.m_conv2d;

    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::make_shared<CNN_layer_op>(
            *(dynamic_cast<CNN_layer_op*>(other.m_ops[i].get())));
    }

    return *this;
}


neurons::CNN_layer & neurons::CNN_layer::operator = (CNN_layer && other)
{
    NN_layer::operator=(other);
    this->m_conv2d = std::move(other.m_conv2d);

    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::move(other.m_ops[i]);
    }

    return *this;
}


neurons::Shape neurons::CNN_layer::output_shape() const
{
    return this->m_conv2d.get_output_shape();
}

std::unique_ptr<char[]> neurons::CNN_layer::to_binary_data(lint & data_size) const
{
    lint size;
    std::unique_ptr<char[]> l_d = Traditional_NN_layer::to_binary_data(size);

    char * layer_data = new char[size + 2 * sizeof(lint)];
    memcpy(layer_data, l_d.get(), size);

    char * position = layer_data + size;
    *(reinterpret_cast<lint*>(position)) = this->m_conv2d.r_stride();
    position += sizeof(lint);
    *(reinterpret_cast<lint*>(position)) = this->m_conv2d.r_zero_p();

    // Do not forget to increase the header size of this layer data
    *(reinterpret_cast<lint*>(layer_data)) += 2 * sizeof(lint);
    
    data_size = size + 2 * sizeof(lint);

    
    return std::unique_ptr<char[]>(layer_data);
}

//////////////////////////////////////////////////
neurons::CNN_layer_op::CNN_layer_op()
{}

neurons::CNN_layer_op::CNN_layer_op(
    const Conv_2d & conv2d,
    const TMatrix<> & w,
    const TMatrix<> & b,
    const std::unique_ptr<Activation>& act_func,
    const std::unique_ptr<ErrorFunction>& err_func)
    :
    Traditional_NN_layer_op(w, b, act_func, err_func),
    m_conv2d{ conv2d }
{}

neurons::CNN_layer_op::CNN_layer_op(const CNN_layer_op & other)
    :
    Traditional_NN_layer_op(other),
    m_conv2d{ other.m_conv2d }
{}

neurons::CNN_layer_op::CNN_layer_op(CNN_layer_op && other)
    :
    Traditional_NN_layer_op(other),
    m_conv2d{ std::move(other.m_conv2d) }
{}

neurons::CNN_layer_op & neurons::CNN_layer_op::operator = (const CNN_layer_op & other)
{
    NN_layer_op::operator = (other);
    this->m_conv2d = other.m_conv2d;

    return *this;
}

neurons::CNN_layer_op & neurons::CNN_layer_op::operator = (CNN_layer_op && other)
{
    NN_layer_op::operator = (other);
    this->m_conv2d = std::move(other.m_conv2d);

    return *this;
}

std::vector<neurons::TMatrix<>> neurons::CNN_layer_op::batch_forward_propagate(const std::vector<TMatrix<>>& inputs)
{
    if (nullptr == this->m_act_func)
    {
        throw std::invalid_argument(
            std::string("neurons::CNN_layer::forward_propagate: activation function is expected, but it does not exist."));
    }

    size_t samples = inputs.size();

    std::vector<neurons::TMatrix<>> outputs{ samples };
    this->m_act_diffs.resize(samples);
    this->m_conv_to_x_diffs.resize(samples);
    this->m_conv_to_w_diffs.resize(samples);

    for (size_t i = 0; i < samples; ++i)
    {
        // Convolutional multiplication of this sample
        TMatrix<> conv_product = this->m_conv2d(inputs[i], this->m_w, this->m_b);

        this->m_conv_to_x_diffs[i] = this->m_conv2d.get_diff_to_input();
        this->m_conv_to_w_diffs[i] = this->m_conv2d.get_diff_to_weights();

        // Execute activation function of this sample
        this->m_act_func->operator()(outputs[i], this->m_act_diffs[i], conv_product);
    }

    // std::cout << outputs[0];
    return outputs;
}

std::vector<neurons::TMatrix<>> neurons::CNN_layer_op::batch_forward_propagate(
    const std::vector<TMatrix<>>& inputs, const std::vector<TMatrix<>>& targets)
{
    if (nullptr == this->m_err_func)
    {
        throw std::invalid_argument(
            std::string("neurons::CNN_layer::forward_propagate: error function is expected, but it does not exist."));
    }

    size_t samples = inputs.size();

    std::vector<neurons::TMatrix<>> outputs{ samples };
    this->m_act_diffs.resize(samples);
    this->m_conv_to_x_diffs.resize(samples);
    this->m_conv_to_w_diffs.resize(samples);

    for (size_t i = 0; i < samples; ++i)
    {
        // Convolutional multiplication of this sample
        TMatrix<> conv_product = this->m_conv2d(inputs[i], this->m_w, this->m_b);

        this->m_conv_to_x_diffs[i] = this->m_conv2d.get_diff_to_input();
        this->m_conv_to_w_diffs[i] = this->m_conv2d.get_diff_to_weights();

        // Execute activation function of this sample
        this->m_loss += this->m_err_func->operator()(outputs[i], this->m_act_diffs[i], targets[i], conv_product);
    }

    return outputs;
}


std::vector<neurons::TMatrix<>> neurons::CNN_layer_op::batch_back_propagate(double l_rate, const std::vector<TMatrix<>> &E_to_y_diffs)
{
    size_t samples = E_to_y_diffs.size();
    std::vector<TMatrix<>> E_to_x_diffs{ samples };

    this->m_w_gradient = 0;
    this->m_b_gradient = 0;

    for (size_t i = 0; i < samples; ++i)
    {
        // Multiply element by element
        // dE/dz = (dy/dz) * (dE/dy)
        neurons::TMatrix<> diff_E_to_z = neurons::multiply(this->m_act_diffs[i], E_to_y_diffs[i]);

        // dE/dx = (dz/dx) * (dE/dz) 
        E_to_x_diffs[i] = neurons::matrix_multiply(this->m_conv_to_x_diffs[i], diff_E_to_z.right_extend_shape(), 3, 4);
        E_to_x_diffs[i].reshape(E_to_x_diffs[i].shape().sub_shape(0, E_to_x_diffs[i].shape().dim() - 2));
        
        // Update weights
        this->m_w_gradient += neurons::matrix_multiply(this->m_conv_to_w_diffs[i], diff_E_to_z, 2, 3);

        // Update the bias
        this->m_b_gradient += diff_E_to_z.reduce_mean(1).reduce_mean(1);
    }

    this->m_w_gradient *= l_rate;
    this->m_b_gradient *= l_rate;

    return E_to_x_diffs;
}


std::vector<neurons::TMatrix<>> neurons::CNN_layer_op::batch_back_propagate(double l_rate)
{
    size_t samples = this->m_conv_to_x_diffs.size();
    std::vector<TMatrix<>> E_to_x_diffs{ samples };

    this->m_w_gradient = 0;
    this->m_b_gradient = 0;

    for (size_t i = 0; i < samples; ++i)
    {
        // dE/dx = (dz/dx) * (dE/dz) 
        E_to_x_diffs[i] = neurons::matrix_multiply(this->m_conv_to_x_diffs[i], this->m_act_diffs[i].right_extend_shape(), 3, 4);
        E_to_x_diffs[i].reshape(E_to_x_diffs[i].shape().sub_shape(0, E_to_x_diffs[i].shape().dim() - 2));

        // Update weights
        this->m_w_gradient += neurons::matrix_multiply(this->m_conv_to_w_diffs[i], this->m_act_diffs[i], 2, 3);

        // Update the bias
        this->m_b_gradient += this->m_act_diffs[i].reduce_mean(1).reduce_mean(1);
    }

    this->m_w_gradient *= l_rate;
    this->m_b_gradient *= l_rate;

    return E_to_x_diffs;
}

neurons::Shape neurons::CNN_layer_op::output_shape() const
{
    return this->m_conv2d.get_output_shape();
}
