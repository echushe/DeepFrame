#include "NN_layer.h"

const std::string neurons::NN_layer::NN{ "NN" };
const std::string neurons::NN_layer::FCNN{ "FCNN" };
const std::string neurons::NN_layer::CNN{ "CNN" };
const std::string neurons::NN_layer::RNN{ "RNN" };

neurons::NN_layer::NN_layer()
{}

neurons::NN_layer::NN_layer(lint threads)
    : m_ops{ static_cast<size_t>(threads) }
{}

neurons::NN_layer::NN_layer(const NN_layer & other)
    : m_ops{ static_cast<size_t>(other.m_ops.size()) }
{}

neurons::NN_layer::NN_layer(NN_layer && other)
    : m_ops{ static_cast<size_t>(other.m_ops.size()) }
{}

neurons::NN_layer & neurons::NN_layer::operator = (const NN_layer & other)
{
    this->m_ops.resize(other.m_ops.size());

    return *this;
}


neurons::NN_layer & neurons::NN_layer::operator = (NN_layer && other)
{
    this->m_ops.resize(other.m_ops.size());

    return *this;
}

std::vector<std::shared_ptr<neurons::NN_layer_op>>& neurons::NN_layer::operation_instances() const
{
    return this->m_ops;
}

double neurons::NN_layer::commit_training()
{
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        loss += this->m_ops[i]->get_loss();
        this->m_ops[i]->clear_loss();
    }

    return loss;
}

double neurons::NN_layer::commit_testing()
{
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        loss += this->m_ops[i]->get_loss();
        this->m_ops[i]->clear_loss();
    }

    return loss;
}

neurons::NN_layer_op::NN_layer_op()
    : m_loss {0}
{}


neurons::NN_layer_op::NN_layer_op(const NN_layer_op & other)
    : m_loss{ other.m_loss }
{}


neurons::NN_layer_op::NN_layer_op(NN_layer_op && other)
    : m_loss{ std::move(other.m_loss) }
{}


neurons::NN_layer_op & neurons::NN_layer_op::operator = (const NN_layer_op & other)
{
    this->m_loss = other.m_loss;

    return *this;
}

neurons::NN_layer_op & neurons::NN_layer_op::operator = (NN_layer_op && other)
{
    this->m_loss = std::move(other.m_loss);

    return *this;
}

double neurons::NN_layer_op::get_loss() const
{
    return this->m_loss;
}

void neurons::NN_layer_op::clear_loss()
{
    this->m_loss = 0;
}

