#include "RNN_unit.h"

neurons::RNN_unit::RNN_unit()
{}

neurons::RNN_unit::RNN_unit(
    lint input_size,
    lint output_size,
    Activation *act_func,
    ErrorFunction *err_func)
    :
    m_u { Shape { output_size, output_size } },
    m_w { Shape { input_size, output_size } },
    m_b { Shape { 1, output_size } },
    m_old_y_in{ Shape{ 1, output_size } },
    m_act_func { act_func },
    m_err_func{ err_func }
{
    this->m_u.gaussian_random(0, 1);
    this->m_w.gaussian_random(0, 1);
    this->m_b.gaussian_random(0, 1);
    this->m_old_y_in.gaussian_random(0, 1);
    this->m_old_y_bak = this->m_old_y_in;
}

neurons::RNN_unit::RNN_unit(
    lint input_size,
    lint output_size,
    std::unique_ptr<neurons::Activation> &act_func,
    std::unique_ptr<neurons::ErrorFunction> &err_func)
    :
    m_u{ Shape{ output_size, output_size } },
    m_w{ Shape{ input_size, output_size } },
    m_b{ Shape{ 1, output_size } },
    m_old_y_in{ Shape{ 1, output_size } },
    m_act_func{ std::move(act_func) },
    m_err_func{ std::move(err_func) }
{
    this->m_u.gaussian_random(0, 1);
    this->m_w.gaussian_random(0, 1);
    this->m_b.gaussian_random(0, 1);
    this->m_old_y_in.gaussian_random(0, 1);
    this->m_old_y_bak = this->m_old_y_in;
}


neurons::RNN_unit::RNN_unit(const RNN_unit & other)
    :
    m_u { other.m_u },
    m_w { other.m_w },
    m_b { other.m_b },
    m_old_y_bak{ other.m_old_y_bak },
    m_old_y_in{ other.m_old_y_in },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr }
{}


neurons::RNN_unit::RNN_unit(RNN_unit && other)
    :
    m_u{ std::move(other.m_u) },
    m_w{ std::move(other.m_w) },
    m_b{ std::move(other.m_b) },
    m_old_y_bak{ std::move(other.m_old_y_bak) },
    m_old_y_in{ std::move(other.m_old_y_in) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) }
{}


neurons::RNN_unit & neurons::RNN_unit::operator = (const RNN_unit & other)
{
    this->m_u = other.m_u;
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_old_y_bak = other.m_old_y_bak;
    this->m_old_y_in = other.m_old_y_in;
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;

    return *this;
}


neurons::RNN_unit & neurons::RNN_unit::operator = (RNN_unit && other)
{
    this->m_u = std::move(other.m_u);
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_old_y_bak = std::move(other.m_old_y_bak);
    this->m_old_y_in = std::move(other.m_old_y_in);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);

    return *this;
}


neurons::Shape neurons::RNN_unit::output_shape() const
{
    return this->m_b.shape();
}


neurons::Matrix neurons::RNN_unit::forward_propagate(const Matrix& input)
{
    if (nullptr == this->m_act_func)
    {
        throw std::invalid_argument(
            std::string("neurons::RNN_layer::forward_propagate: activation function is expected, but it does not exist."));
    }

    this->m_x = input;
    
    // z = old_y * w + x * w + b
    neurons::Matrix product = this->m_old_y_in * this->m_u + this->m_x * this->m_w + this->m_b;
    // y = g(z)
    this->m_act_func->operator()(this->m_y, this->m_act_diffs, product);
    this->m_old_y_bak = this->m_old_y_in;
    this->m_old_y_in = this->m_y;

    return this->m_y;
}


neurons::Matrix neurons::RNN_unit::forward_propagate(double &loss, const Matrix &input, const Matrix &targets)
{
    if (nullptr == this->m_err_func)
    {
        throw std::invalid_argument(
            std::string("neurons::RNN_layer::forward_propagate: error function is expected, but it does not exist."));
    }

    this->m_x = input;
    neurons::Matrix output;

    // z = old_y * u + x * w + b
    neurons::Matrix product = this->m_old_y_in * this->m_u + this->m_x * this->m_w + this->m_b;
    // y = g(z) and E = error(y, t)
    loss = this->m_err_func->operator()(this->m_y, this->m_act_diffs, targets, product);
    this->m_old_y_bak = this->m_old_y_in;
    this->m_old_y_in = this->m_y;

    return this->m_y;
}


neurons::Matrix neurons::RNN_unit::backward_propagate(double l_rate, const Matrix &E_to_y_diffs)
{
    Matrix E_to_x_diffs;

    neurons::Matrix diff_E_to_z = neurons::multiply(this->m_act_diffs, neurons::transpose(E_to_y_diffs));

    // Calculate the derivative dE/dx via the chain rule (back propagation).
    // E is the error from the last layer.
    // x is input of the current layer.
    E_to_x_diffs = this->m_w * neurons::transpose(diff_E_to_z);

    // Calculate dE/du and update the weights.
    // E is the error from the last layer.
    // u are weights of the current layer.
    Matrix u_gradient = neurons::transpose(this->m_old_y_bak) * diff_E_to_z;

    // Calculate dE/dw and update the weights.
    // E is the error from the last layer.
    // w are weights of the current layer.
    Matrix w_gradient = neurons::transpose(this->m_x) * diff_E_to_z;

    // Calculate dE/db and update the bias.
    // E is the error from the last layer.
    // w are bias of the current layer.
    Matrix b_gradient = diff_E_to_z;

    this->m_old_y_bak = this->m_y;

    this->m_u -= u_gradient * l_rate;
    this->m_w -= w_gradient * l_rate;
    this->m_b -= b_gradient * l_rate;

    return E_to_x_diffs;
}


neurons::Matrix neurons::RNN_unit::backward_propagate(double l_rate)
{
    Matrix E_to_x_diffs;

    // Calculate the derivative dE/dx via the chain rule (back propagation).
    // E is the error from the last layer.
    // x is input of the current layer.
    E_to_x_diffs = this->m_w * neurons::transpose(this->m_act_diffs);

    // Calculate dE/du and update the weights.
    // E is the error from the last layer.
    // u are weights of the current layer.
    Matrix u_gradient = neurons::transpose(this->m_old_y_bak) * this->m_act_diffs;

    // Calculate dE/dw and update the weights.
    // E is the error from the last layer.
    // w are weights of the current layer.
    Matrix w_gradient = neurons::transpose(this->m_x) * this->m_act_diffs;

    // Calculate dE/db and update the bias.
    // E is the error from the last layer.
    // w are bias of the current layer.
    Matrix b_gradient = this->m_act_diffs;

    this->m_old_y_bak = this->m_y;

    this->m_u -= u_gradient * l_rate;
    this->m_w -= w_gradient * l_rate;
    this->m_b -= b_gradient * l_rate;

    return E_to_x_diffs;
}


