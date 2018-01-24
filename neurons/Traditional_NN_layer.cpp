#include "Traditional_NN_layer.h"

neurons::Traditional_NN_layer::Traditional_NN_layer()
{}

neurons::Traditional_NN_layer::Traditional_NN_layer(
    const Shape & w_sh,
    const Shape & b_sh,
    lint threads,
    Activation *act_func,
    ErrorFunction *err_func)
    :
    NN_layer( threads ),
    m_w{ w_sh },
    m_b{ b_sh },
    m_act_func{ act_func },
    m_err_func{ err_func }
{
    this->m_w.gaussian_random(0, 0.1);
    this->m_b.gaussian_random(0, 0.1);
}

neurons::Traditional_NN_layer::Traditional_NN_layer(const Traditional_NN_layer & other)
    : 
    NN_layer(other),
    m_w{ other.m_w }, m_b{ other.m_b },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr }
{}

neurons::Traditional_NN_layer::Traditional_NN_layer(Traditional_NN_layer && other)
    : 
    NN_layer(other),
    m_w{ std::move(other.m_w) }, m_b{ std::move(other.m_b) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) }
{}

neurons::Traditional_NN_layer & neurons::Traditional_NN_layer::operator = (const Traditional_NN_layer & other)
{
    NN_layer::operator=(other);
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;

    return *this;
}


neurons::Traditional_NN_layer & neurons::Traditional_NN_layer::operator = (Traditional_NN_layer && other)
{
    NN_layer::operator=(other);
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);

    return *this;
}

double neurons::Traditional_NN_layer::commit_training()
{
    Matrix w_gradient_sum{ this->m_w.shape(), 0 };
    Matrix b_gradient_sum{ this->m_b.shape(), 0 };
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        auto op = dynamic_cast<Traditional_NN_layer_op*>(this->m_ops[i].get());
        w_gradient_sum += op->get_weight_gradient();
        b_gradient_sum += op->get_bias_gradient();
        loss += this->m_ops[i]->get_loss();
        this->m_ops[i]->clear_loss();
    }

    this->m_w -= w_gradient_sum;
    this->m_b -= b_gradient_sum;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        auto op = dynamic_cast<Traditional_NN_layer_op*>(this->m_ops[i].get());
        op->update_w_and_b(this->m_w, this->m_b);
    }

    return loss;
}

double neurons::Traditional_NN_layer::commit_testing()
{
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        loss += this->m_ops[i]->get_loss();
        this->m_ops[i]->clear_loss();
    }

    return loss;
}

neurons::Traditional_NN_layer_op::Traditional_NN_layer_op(
    const Matrix & w,
    const Matrix & b,
    const std::unique_ptr<Activation> &act_func,
    const std::unique_ptr<ErrorFunction> &err_func)
    :
    NN_layer_op(),
    m_w{ w }, m_b{ b },
    m_act_func{ act_func ? act_func->clone() : nullptr },
    m_err_func{ err_func ? err_func->clone() : nullptr },
    m_w_gradient{ m_w.shape(), 0 },
    m_b_gradient{ m_b.shape(), 0 }
{
    this->m_w.gaussian_random(0, 0.1);
    this->m_b.gaussian_random(0, 0.1);
}


neurons::Traditional_NN_layer_op::Traditional_NN_layer_op(const Traditional_NN_layer_op & other)
    :
    NN_layer_op(other),
    m_w{ other.m_w },
    m_b{ other.m_b },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr },
    m_w_gradient{ other.m_w_gradient },
    m_b_gradient{ other.m_b_gradient },
    m_act_diffs{ other.m_act_diffs }
{}


neurons::Traditional_NN_layer_op::Traditional_NN_layer_op(Traditional_NN_layer_op && other)
    :
    NN_layer_op(other),
    m_w{ std::move(other.m_w) },
    m_b{ std::move(other.m_b) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) },
    m_w_gradient{ std::move(other.m_w_gradient) },
    m_b_gradient{ std::move(other.m_b_gradient) },
    m_act_diffs{ std::move(other.m_act_diffs) }
{}


neurons::Traditional_NN_layer_op & neurons::Traditional_NN_layer_op::operator = (const Traditional_NN_layer_op & other)
{
    NN_layer_op::operator=(other);
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;
    this->m_w_gradient = other.m_w_gradient;
    this->m_b_gradient = other.m_b_gradient;
    this->m_act_diffs = other.m_act_diffs;

    return *this;
}

neurons::Traditional_NN_layer_op & neurons::Traditional_NN_layer_op::operator = (Traditional_NN_layer_op && other)
{
    NN_layer_op::operator=(other);
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);
    this->m_w_gradient = std::move(other.m_w_gradient);
    this->m_b_gradient = std::move(other.m_b_gradient);
    this->m_act_diffs = std::move(other.m_act_diffs);

    return *this;
}

neurons::Matrix neurons::Traditional_NN_layer_op::forward_propagate(const Matrix & input)
{
    std::vector<Matrix> inputs;
    inputs.push_back(input);

    return this->batch_forward_propagate(inputs)[0];
}

neurons::Matrix neurons::Traditional_NN_layer_op::forward_propagate(const Matrix & input, const Matrix & target)
{
    std::vector<Matrix> inputs, targets;
    inputs.push_back(input);
    targets.push_back(target);

    return this->batch_forward_propagate(inputs, targets)[0];
}

neurons::Matrix neurons::Traditional_NN_layer_op::backward_propagate(double l_rate, const Matrix & E_to_y_diff)
{
    std::vector<Matrix> E_to_y_diffs;
    E_to_y_diffs.push_back(E_to_y_diff);

    return this->batch_backward_propagate(l_rate, E_to_y_diffs)[0];
}

neurons::Matrix neurons::Traditional_NN_layer_op::backward_propagate(double l_rate)
{
    return this->batch_backward_propagate(l_rate)[0];
}

neurons::Matrix& neurons::Traditional_NN_layer_op::get_weight_gradient() const
{
    return this->m_w_gradient;
}

neurons::Matrix& neurons::Traditional_NN_layer_op::get_bias_gradient() const
{
    return this->m_b_gradient;
}

void neurons::Traditional_NN_layer_op::update_w_and_b(const Matrix & w, const Matrix & b)
{
    this->m_w = w;
    this->m_b = b;
}