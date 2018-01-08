#include "NN_layer.h"

neurons::NN_layer::NN_layer()
{}

neurons::NN_layer::NN_layer(
    const Shape & w_sh,
    const Shape & b_sh,
    lint threads,
    Activation *act_func,
    ErrorFunction *err_func)
    :
    m_w{w_sh},
    m_b{b_sh},
    m_act_func{act_func},
    m_err_func{err_func},
    m_ops{ static_cast<size_t>(threads) }
{
    this->m_w.gaussian_random(0, 0.1);
    this->m_b.gaussian_random(0, 0.1);
}

neurons::NN_layer::NN_layer(const NN_layer & other)
    : m_w{ other.m_w }, m_b{ other.m_b },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr }
{}

neurons::NN_layer::NN_layer(NN_layer && other)
    : m_w{ std::move(other.m_w) }, m_b{ std::move(other.m_b) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) }
{}

neurons::NN_layer & neurons::NN_layer::operator = (const NN_layer & other)
{
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;

    return *this;
}


neurons::NN_layer & neurons::NN_layer::operator = (NN_layer && other)
{
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);

    return *this;
}

std::vector<std::shared_ptr<neurons::NN_layer_op>>& neurons::NN_layer::operation_instances() const
{
    return this->m_ops;
}


double neurons::NN_layer::commit_training()
{
    Matrix w_gradient_sum{ this->m_w.shape(), 0 };
    Matrix b_gradient_sum{ this->m_b.shape(), 0 };
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        w_gradient_sum += this->m_ops[i]->get_weight_gradient();
        b_gradient_sum += this->m_ops[i]->get_bias_gradient();
        loss += this->m_ops[i]->get_loss();
    }

    this->m_w -= w_gradient_sum;
    this->m_b -= b_gradient_sum;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        this->m_ops[i]->update_w_and_b(this->m_w, this->m_b);
    }

    return loss;
}

double neurons::NN_layer::commit_testing()
{
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        loss += this->m_ops[i]->get_loss();
    }

    return loss;
}

neurons::NN_layer_op::NN_layer_op(
    const Matrix & w,
    const Matrix & b,
    const std::unique_ptr<Activation> &act_func,
    const std::unique_ptr<ErrorFunction> &err_func)
    : m_w{w}, m_b{b},
    m_act_func{ act_func ? act_func->clone() : nullptr },
    m_err_func{ err_func ? err_func->clone() : nullptr },
    m_w_gradient{ m_w.shape(), 0 },
    m_b_gradient{ m_b.shape(), 0 },
    m_loss {0}
{
    this->m_w.gaussian_random(0, 0.1);
    this->m_b.gaussian_random(0, 0.1);
}


neurons::NN_layer_op::NN_layer_op(const NN_layer_op & other)
    : m_w{ other.m_w }, m_b{ other.m_b },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr },
    m_w_gradient{ other.m_w_gradient },
    m_b_gradient{ other.m_b_gradient },
    m_act_diffs{ other.m_act_diffs }
{}


neurons::NN_layer_op::NN_layer_op(NN_layer_op && other)
    : m_w{ std::move(other.m_w) }, m_b{ std::move(other.m_b) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) },
    m_w_gradient{ std::move(other.m_w_gradient) },
    m_b_gradient{ std::move(other.m_b_gradient) },
    m_act_diffs{ std::move(other.m_act_diffs) }
{}


neurons::NN_layer_op & neurons::NN_layer_op::operator = (const NN_layer_op & other)
{
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;
    this->m_w_gradient = other.m_w_gradient;
    this->m_b_gradient = other.m_b_gradient;
    this->m_act_diffs = other.m_act_diffs;

    return *this;
}

neurons::NN_layer_op & neurons::NN_layer_op::operator = (NN_layer_op && other)
{
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);
    this->m_w_gradient = std::move(other.m_w_gradient);
    this->m_b_gradient = std::move(other.m_b_gradient);
    this->m_act_diffs = std::move(other.m_act_diffs);

    return *this;
}

neurons::Matrix& neurons::NN_layer_op::get_weight_gradient() const
{
    return this->m_w_gradient;
}

neurons::Matrix& neurons::NN_layer_op::get_bias_gradient() const
{
    return this->m_b_gradient;
}

double neurons::NN_layer_op::get_loss() const
{
    return this->m_loss;
}

void neurons::NN_layer_op::update_w_and_b(const Matrix & w, const Matrix & b)
{
    this->m_w = w;
    this->m_b = b;
}

