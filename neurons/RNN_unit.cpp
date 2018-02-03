#include "RNN_unit.h"

neurons::RNN_unit::RNN_unit()
{}

neurons::RNN_unit::RNN_unit(
    lint input_size,
    lint output_size,
    lint bptt_len,
    Activation *act_func,
    ErrorFunction *err_func)
    :
    m_u { Shape { output_size, output_size } },
    m_w { Shape { input_size, output_size } },
    m_b { Shape { 1, output_size } },
    m_old_y{ Shape{ 1, output_size }, 0 },
    m_act_func { act_func },
    m_err_func { err_func },
    m_bptt_len { bptt_len }
{
    this->m_u.gaussian_random(0, 0.01);
    this->m_w.gaussian_random(0, 0.01);
    this->m_b.gaussian_random(0, 0.01);
}

neurons::RNN_unit::RNN_unit(
    lint input_size,
    lint output_size,
    lint bptt_len,
    std::unique_ptr<neurons::Activation> &act_func,
    std::unique_ptr<neurons::ErrorFunction> &err_func)
    :
    m_u{ Shape{ output_size, output_size } },
    m_w{ Shape{ input_size, output_size } },
    m_b{ Shape{ 1, output_size } },
    m_old_y{ Shape{ 1, output_size }, 0 },
    m_act_func{ std::move(act_func) },
    m_err_func{ std::move(err_func) },
    m_bptt_len{ bptt_len }
{
    this->m_u.gaussian_random(0, 0.01);
    this->m_w.gaussian_random(0, 0.01);
    this->m_b.gaussian_random(0, 0.01);
}


neurons::RNN_unit::RNN_unit(const RNN_unit & other)
    :
    m_u { other.m_u },
    m_w { other.m_w },
    m_b { other.m_b },
    m_old_y{ other.m_old_y },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr },
    m_bptt_len{ other.m_bptt_len }
{}


neurons::RNN_unit::RNN_unit(RNN_unit && other)
    :
    m_u{ std::move(other.m_u) },
    m_w{ std::move(other.m_w) },
    m_b{ std::move(other.m_b) },
    m_old_y{ std::move(other.m_old_y) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) },
    m_bptt_len{ other.m_bptt_len }
{}


neurons::RNN_unit & neurons::RNN_unit::operator = (const RNN_unit & other)
{
    this->m_u = other.m_u;
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_old_y = other.m_old_y;
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;
    this->m_bptt_len = other.m_bptt_len;

    return *this;
}


neurons::RNN_unit & neurons::RNN_unit::operator = (RNN_unit && other)
{
    this->m_u = std::move(other.m_u);
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_old_y = std::move(other.m_old_y);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);
    this->m_bptt_len = other.m_bptt_len;

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
    
    // z = old_y * w + x * w + b
    neurons::Matrix product = this->m_old_y * this->m_u + input * this->m_w + this->m_b;
    // y = g(z)
    Matrix new_y;
    // Differentiation of the activation function dy/dz
    // in which y is output of activation, z is x * w  + b
    Matrix act_diff;
    this->m_act_func->operator()(new_y, act_diff, product);

    if (this->m_cache_for_bptt.size() == this->m_bptt_len)
    {
        this->m_cache_for_bptt.pop_front();
    }

    this->m_cache_for_bptt.push_back(cache_item{this->m_old_y, input, act_diff });
    this->m_old_y = new_y;

    // std::cout << input;

    if (this->m_counter < 1000000)
    {
        ++this->m_counter;
    }
    else
    {
        std::cout << this->m_u;
        std::cout << this->m_w;
        std::cout << product;
        std::cout << new_y;
        std::cout << act_diff;
        this->m_counter = 0;
    }

    return new_y;
}


neurons::Matrix neurons::RNN_unit::forward_propagate(double &loss, const Matrix &input, const Matrix &targets)
{
    if (nullptr == this->m_err_func)
    {
        throw std::invalid_argument(
            std::string("neurons::RNN_layer::forward_propagate: error function is expected, but it does not exist."));
    }

    neurons::Matrix output;

    // z = old_y * u + x * w + b
    neurons::Matrix product = this->m_old_y * this->m_u + input * this->m_w + this->m_b;
    // y = g(z) and E = error(y, t)
    Matrix new_y;
    // Differentiation of the activation function dy/dz
    // in which y is output of activation, z is x * w  + b
    Matrix act_diff;
    loss = this->m_err_func->operator()(new_y, act_diff, targets, product);

    if (this->m_cache_for_bptt.size() == this->m_bptt_len)
    {
        this->m_cache_for_bptt.pop_front();
    }

    this->m_cache_for_bptt.push_back(cache_item{this->m_old_y, input, act_diff });
    this->m_old_y = new_y;

    return new_y;
}


std::vector<neurons::Matrix> neurons::RNN_unit::backward_propagate_through_time(
    double l_rate, const Matrix &E_to_y_diff, lint len)
{
    if (0 == len)
    {
        len = this->m_bptt_len;
    }

    Matrix u_gradient_sum{ this->m_u.shape(), 0 };
    Matrix w_gradient_sum{ this->m_w.shape(), 0 };
    Matrix b_gradient_sum{ this->m_b.shape(), 0 };

    std::vector<Matrix> E_to_x_diffs;
    Matrix E_to_old_y_diff = E_to_y_diff;

    // Back propagation through time (BPTT)
    for (lint i = this->m_cache_for_bptt.size() - 1; i >=0; --i)
    {
        cache_item & it = this->m_cache_for_bptt[i];

        neurons::Matrix diff_E_to_z = neurons::multiply(it.m_act_diff, neurons::transpose(E_to_old_y_diff));

        // std::cout << E_to_old_y_diff;

        // Calculate the derivative dE/d(old_y) via the chain rule (back propagation).
        // E is the error from the last layer.
        // old_y is output of last time as input of the current layer.
        E_to_old_y_diff = this->m_u * neurons::transpose(diff_E_to_z);

        // Calculate the derivative dE/dx via the chain rule (back propagation).
        // E is the error from the last layer.
        // x is input of the current layer.
        Matrix E_to_x_diff = this->m_w * neurons::transpose(diff_E_to_z);

        // Calculate dE/du and update the weights.
        // E is the error from the last layer.
        // u are weights of the current layer.
        Matrix u_gradient = neurons::transpose(it.m_y_in) * diff_E_to_z;

        // Calculate dE/dw and update the weights.
        // E is the error from the last layer.
        // w are weights of the current layer.
        Matrix w_gradient = neurons::transpose(it.m_x) * diff_E_to_z;

        // Calculate dE/db and update the bias.
        // E is the error from the last layer.
        // w are bias of the current layer.
        Matrix b_gradient = diff_E_to_z;

        u_gradient_sum += u_gradient;
        w_gradient_sum += w_gradient;
        b_gradient_sum += b_gradient;

        E_to_x_diffs.push_back(E_to_x_diff);

        --len;
        if (0 == len)
        {
            break;
        }
    }

    u_gradient_sum /= this->m_cache_for_bptt.size();
    w_gradient_sum /= this->m_cache_for_bptt.size();
    b_gradient_sum /= this->m_cache_for_bptt.size();

    this->m_u -= u_gradient_sum * l_rate;
    this->m_w -= w_gradient_sum * l_rate;
    this->m_b -= b_gradient_sum * l_rate;

    // std::cout << "===================================================================================\n";
    // std::cout << u_gradient_sum;
    // std::cout << w_gradient_sum;

    return E_to_x_diffs;
}


std::vector<neurons::Matrix> neurons::RNN_unit::backward_propagate_through_time(double l_rate, lint len)
{
    Matrix E_to_y_diff{ this->m_b.shape(), 1 };  

    return this->backward_propagate_through_time(l_rate, E_to_y_diff, len);
}

void neurons::RNN_unit::forget_all()
{
    this->m_old_y = 0;
}



