#include "FCNN_layer.h"

neurons::FCNN_layer::FCNN_layer()
{}

neurons::FCNN_layer::FCNN_layer(
    lint input_size,
    lint output_size,
    lint threads,
    neurons::Activation *act_func,
    neurons::ErrorFunction *err_func)
    :
    Traditional_NN_layer(neurons::Shape{ input_size, output_size }, neurons::Shape{ 1, output_size }, threads, act_func, err_func)
{
    double var = static_cast<double>(10) / this->m_w.shape()[0];
    this->m_w.gaussian_random(0, var);
    this->m_b.gaussian_random(0, var);

    for (lint i = 0; i < threads; ++i)
    {
        this->m_ops[i] = std::make_shared<FCNN_layer_op>( this->m_w, this->m_b, this->m_act_func, this->m_err_func );
    }
}

neurons::FCNN_layer::FCNN_layer(const FCNN_layer & other)
    : Traditional_NN_layer(other)
{
    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::make_shared<FCNN_layer_op>(
            *(dynamic_cast<FCNN_layer_op*>(other.m_ops[i].get())));
    }
}

neurons::FCNN_layer::FCNN_layer(FCNN_layer && other)
    : Traditional_NN_layer(other)
{
    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::move(other.m_ops[i]);
    }
}

neurons::FCNN_layer & neurons::FCNN_layer::operator=(const FCNN_layer & other)
{
    NN_layer::operator=(other);

    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::make_shared<FCNN_layer_op>(
            *(dynamic_cast<FCNN_layer_op*>(other.m_ops[i].get())));
    }

    return *this;
}

neurons::FCNN_layer & neurons::FCNN_layer::operator=(FCNN_layer && other)
{
    NN_layer::operator=(other);

    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::move(other.m_ops[i]);
    }

    return *this;
}

neurons::Shape neurons::FCNN_layer::output_shape() const
{
    return this->m_b.shape();
}

/////////////////////////////////////////////////

neurons::FCNN_layer_op::FCNN_layer_op()
{}

neurons::FCNN_layer_op::FCNN_layer_op(
    const Matrix &w,
    const Matrix &b,
    const std::unique_ptr<Activation> &act_func,
    const std::unique_ptr<ErrorFunction> &err_func)
    : Traditional_NN_layer_op(w, b, act_func, err_func)
{}

neurons::FCNN_layer_op::FCNN_layer_op(const FCNN_layer_op & other)
    : Traditional_NN_layer_op(other)
{}

neurons::FCNN_layer_op::FCNN_layer_op(FCNN_layer_op && other)
    : Traditional_NN_layer_op(other)
{}

neurons::FCNN_layer_op & neurons::FCNN_layer_op::operator = (const FCNN_layer_op & other)
{
    NN_layer_op::operator = (other);
    return *this;
}

neurons::FCNN_layer_op & neurons::FCNN_layer_op::operator = (FCNN_layer_op && other)
{
    NN_layer_op::operator = (other);
    return *this;
}

std::vector<neurons::Matrix> neurons::FCNN_layer_op::batch_forward_propagate(const std::vector<Matrix> & inputs)
{
    if (nullptr == this->m_act_func)
    {
        throw std::invalid_argument(
            std::string("neurons::FCNN_layer::forward_propagate: activation function is expected, but it does not exist."));
    }

    size_t samples = inputs.size();
    this->m_x = inputs;
    std::vector<neurons::Matrix> outputs{ samples };
    this->m_act_diffs.resize(samples);

    for (size_t i = 0; i < samples; ++i)
    {
        // z = x * w + b
        neurons::Matrix product = this->m_x[i] * this->m_w + this->m_b;
        // y = g(z)
        this->m_act_func->operator()(outputs[i], this->m_act_diffs[i], product);
    }

    return outputs;
}


std::vector<neurons::Matrix> neurons::FCNN_layer_op::batch_forward_propagate(
    const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets)
{
    if (nullptr == this->m_err_func)
    {
        throw std::invalid_argument(
            std::string("neurons::FCNN_layer::forward_propagate: error function is expected, but it does not exist."));
    }

    size_t samples = inputs.size();
    this->m_x = inputs;
    std::vector<neurons::Matrix> outputs{ samples };
    this->m_act_diffs.resize(samples);

    for (size_t i = 0; i < samples; ++i)
    {
        // z = x * w + b
        neurons::Matrix product = this->m_x[i] * this->m_w + this->m_b;
        // y = g(z) and E = error(y, t)
        this->m_loss += this->m_err_func->operator()(outputs[i], this->m_act_diffs[i], targets[i], product);
    }

    return outputs;
}


std::vector<neurons::Matrix> neurons::FCNN_layer_op::batch_back_propagate(double l_rate, const std::vector<Matrix> &E_to_y_diffs)
{
    size_t samples = this->m_x.size();
    std::vector<Matrix> E_to_x_diffs{ samples };

    this->m_w_gradient = 0;
    this->m_b_gradient = 0;

    // batch learning of the chain rule (back propagation).
    for (size_t i = 0; i < samples; ++i)
    {
        // Back propagate from y = g(z) to z
        neurons::Matrix diff_E_to_z = neurons::multiply(this->m_act_diffs[i], neurons::transpose(E_to_y_diffs[i]));

        // Calculate the derivative dE/dx
        // E is the error from the last layer.
        // x is input of the current layer.
        E_to_x_diffs[i] = this->m_w * neurons::transpose(diff_E_to_z);

        // Calculate dE/dw and update the weights.
        // E is the error from the last layer.
        // w are weights of the current layer.
        this->m_w_gradient += neurons::transpose(this->m_x[i]) * diff_E_to_z;

        // Calculate dE/db and update the bias.
        // E is the error from the last layer.
        // b are bias of the current layer.
        this->m_b_gradient += diff_E_to_z;
    }

    this->m_w_gradient *= l_rate;
    this->m_b_gradient *= l_rate;

    return E_to_x_diffs;
}


std::vector<neurons::Matrix> neurons::FCNN_layer_op::batch_back_propagate(double l_rate)
{
    size_t samples = this->m_x.size();
    std::vector<Matrix> E_to_x_diffs{ samples };

    this->m_w_gradient = 0;
    this->m_b_gradient = 0;

    // batch learning of the chain rule (back propagation).
    for (size_t i = 0; i < samples; ++i)
    {
        // Calculate the derivative dE/dx
        // E is the error from the last layer.
        // x is input of the current layer.
        E_to_x_diffs[i] = this->m_w * neurons::transpose(this->m_act_diffs[i]);

        // Calculate dE/dw and update the weights.
        // E is the error from the last layer.
        // w are weights of the current layer.
        this->m_w_gradient += neurons::transpose(this->m_x[i]) * this->m_act_diffs[i];

        // Calculate dE/db and update the bias.
        // E is the error from the last layer.
        // b are bias of the current layer.
        this->m_b_gradient += this->m_act_diffs[i];
    }
    
    this->m_w_gradient *= l_rate;
    this->m_b_gradient *= l_rate;

    return E_to_x_diffs;
}

neurons::Shape neurons::FCNN_layer_op::output_shape() const
{
    return this->m_b.shape();
}


