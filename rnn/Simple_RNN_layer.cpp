#include "Simple_RNN_layer.h"

neurons::Simple_RNN_layer::Simple_RNN_layer()
{}

neurons::Simple_RNN_layer::Simple_RNN_layer(
    lint input_size,
    lint output_size,
    lint bptt_len,
    lint threads,
    neurons::Activation *act_func,
    neurons::ErrorFunction *err_func)
    :
    NN_layer(threads),
    m_output_size {output_size},
    m_act_func {act_func},
    m_err_func {err_func}
{
    for (lint i = 0; i < threads; ++i)
    {
        this->m_ops[i] = std::make_shared<Simple_RNN_layer_op>(
            input_size, output_size, bptt_len, this->m_act_func, this->m_err_func);
    }
}

neurons::Simple_RNN_layer::Simple_RNN_layer(const Simple_RNN_layer & other)
    :
    NN_layer(other),
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr }
{
    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::make_shared<Simple_RNN_layer_op>(
            *(dynamic_cast<Simple_RNN_layer_op*>(other.m_ops[i].get())));
    }
}

neurons::Simple_RNN_layer::Simple_RNN_layer(Simple_RNN_layer && other)
    : 
    NN_layer(other),
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) }
{
    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::move(other.m_ops[i]);
    }
}

neurons::Simple_RNN_layer & neurons::Simple_RNN_layer::operator=(const Simple_RNN_layer & other)
{
    NN_layer::operator=(other);
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;

    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::make_shared<Simple_RNN_layer_op>(
            *(dynamic_cast<Simple_RNN_layer_op*>(other.m_ops[i].get())));
    }

    return *this;
}

neurons::Simple_RNN_layer & neurons::Simple_RNN_layer::operator=(Simple_RNN_layer && other)
{
    NN_layer::operator=(other);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);

    for (size_t i = 0; i < other.m_ops.size(); ++i)
    {
        this->m_ops[i] = std::move(other.m_ops[i]);
    }

    return *this;
}

neurons::Shape neurons::Simple_RNN_layer::output_shape() const
{
    return Shape{ 1, this->m_output_size };
}

/////////////////////////////////////////////////

neurons::Simple_RNN_layer_op::Simple_RNN_layer_op()
{}

neurons::Simple_RNN_layer_op::Simple_RNN_layer_op(
    lint input_size,
    lint output_size,
    lint bptt_len,
    const std::unique_ptr<Activation> &act_func,
    const std::unique_ptr<ErrorFunction> &err_func)
    :
    NN_layer_op(),
    m_rnn{
    input_size,
    output_size,
    bptt_len,
    act_func ? act_func->clone() : nullptr,
    err_func ? err_func->clone() : nullptr }
{}

neurons::Simple_RNN_layer_op::Simple_RNN_layer_op(const Simple_RNN_layer_op & other)
    :
    NN_layer_op(other),
    m_rnn{ other.m_rnn }
{}

neurons::Simple_RNN_layer_op::Simple_RNN_layer_op(Simple_RNN_layer_op && other)
    :
    NN_layer_op(other),
    m_rnn{ std::move(other.m_rnn) }
{}

neurons::Simple_RNN_layer_op & neurons::Simple_RNN_layer_op::operator = (const Simple_RNN_layer_op & other)
{
    NN_layer_op::operator = (other);
    this->m_rnn = other.m_rnn;

    return *this;
}

neurons::Simple_RNN_layer_op & neurons::Simple_RNN_layer_op::operator = (Simple_RNN_layer_op && other)
{
    NN_layer_op::operator = (other);
    this->m_rnn = std::move(other.m_rnn);

    return *this;
}

void neurons::Simple_RNN_layer_op::forget_all()
{
    this->m_rnn.forget_all();
}

neurons::Matrix neurons::Simple_RNN_layer_op::forward_propagate(const Matrix & input)
{
    this->m_samples = 1;
    return this->m_rnn.forward_propagate(input);
}

neurons::Matrix neurons::Simple_RNN_layer_op::forward_propagate(const Matrix & input, const Matrix & target)
{
    this->m_samples = 1;
    double loss;

    Matrix pred = this->m_rnn.forward_propagate(loss, input, target);

    this->m_loss += loss;
    return pred;
}

neurons::Matrix neurons::Simple_RNN_layer_op::back_propagate(double l_rate, const Matrix & E_to_y_diff)
{
    return this->m_rnn.back_propagate_through_time(l_rate, E_to_y_diff, 1)[0];
}

neurons::Matrix neurons::Simple_RNN_layer_op::back_propagate(double l_rate)
{
    return this->m_rnn.back_propagate_through_time(l_rate, 1)[0];
}

std::vector<neurons::Matrix> neurons::Simple_RNN_layer_op::batch_forward_propagate(const std::vector<Matrix>& inputs)
{
    size_t size = inputs.size();
    this->m_samples = size;

    std::vector<neurons::Matrix> preds{ size };

    for (size_t i = 0; i < size; ++i)
    {
        preds[i] = this->m_rnn.forward_propagate(inputs[i]);
    }

    return preds;
}


std::vector<neurons::Matrix> neurons::Simple_RNN_layer_op::batch_forward_propagate(
    const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets)
{
    double loss;
    size_t size = inputs.size();
    this->m_samples = size;

    std::vector<neurons::Matrix> preds{ size };

    for (size_t i = 0; i < size; ++i)
    {
        preds[i] = this->m_rnn.forward_propagate(loss, inputs[i], targets[i]);
        this->m_loss += loss;
    }

    return preds;
}


std::vector<neurons::Matrix> neurons::Simple_RNN_layer_op::batch_back_propagate(double l_rate, const std::vector<Matrix> &E_to_y_diffs)
{
    return  this->m_rnn.back_propagate_through_time(l_rate, E_to_y_diffs.back());
}


std::vector<neurons::Matrix> neurons::Simple_RNN_layer_op::batch_back_propagate(double l_rate)
{
    return  this->m_rnn.back_propagate_through_time(l_rate);
}

neurons::Shape neurons::Simple_RNN_layer_op::output_shape() const
{
    return this->m_rnn.output_shape();
}

