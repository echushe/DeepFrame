#include "Traditional_NN_layer.h"


neurons::Traditional_NN_layer::Traditional_NN_layer()
{}

neurons::Traditional_NN_layer::Traditional_NN_layer(
    double mmt_rate, lint threads, const TMatrix<>& w, const TMatrix<>& b,
    std::unique_ptr<Activation>& act_func, std::unique_ptr<ErrorFunction>& err_func)
    :
    NN_layer( threads ),
    m_mmt_rate{ mmt_rate },
    m_w{ w }, m_b{ b },
    m_w_mmt{ w.shape(), 0 }, m_b_mmt{ b.shape(), 0 },
    m_act_func{ std::move(act_func) },
    m_err_func{ std::move(err_func) }
{
    if (nullptr != m_err_func)
    {
        this->m_act_func = this->m_err_func->get_act_func();
    }
}

/*
Structure of a neural network layer:
<
<size of this layer, 64 bit>

<layer type len, layer name>

<weight matrix data>
<bias matrix data>

<act function name len, 8 bit><act function name>

<error function name len, 8 bit><error function name>
>
*/

std::string neurons::Traditional_NN_layer::from_binary_data(
    char * binary_data, lint & data_size, TMatrix<> & w, TMatrix<> & b,
    std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func,
    char *& residual_data, lint & residual_len
)
{
    char * position = binary_data;
    lint size = *(reinterpret_cast<lint *>(position));
    lint real_size = 0;
    
    // Get NN type from binary data
    position += sizeof(lint);
    uint8_t len = *(reinterpret_cast<uint8_t *>(position));
    position += sizeof(uint8_t);
    std::string nn_type{ position, len };

    real_size += sizeof(uint8_t) + len;

    // Get weight matrix from binary data
    position += len;
    lint w_size;
    w = TMatrix<>{ position, w_size };

    real_size += w_size;

    // Get bias matrix from binary data
    position += w_size;
    lint b_size;
    b = TMatrix<>{ position, b_size };

    real_size += b_size;

    // Get activation function
    position += b_size;
    len = *(reinterpret_cast<uint8_t *>(position));

    position += sizeof(uint8_t);
    std::string act_name{ position, len };
    act_func = Activation::get_function_by_name(act_name);

    real_size += sizeof(uint8_t) + len;
    
    // Get error function
    position += len;
    len = *(reinterpret_cast<uint8_t *>(position));

    position += sizeof(uint8_t);
    std::string err_name{ position, len };
    err_func = ErrorFunction::get_function_by_name(err_name);

    real_size += sizeof(uint8_t) + len;

    position += len;
    residual_data = position;
    residual_len = size - real_size;

    data_size = size + sizeof(lint);

    return nn_type;
}


neurons::Traditional_NN_layer::Traditional_NN_layer(
    double mmt_rate,
    const Shape & w_sh,
    const Shape & b_sh,
    lint threads,
    Activation *act_func,
    ErrorFunction *err_func)
    :
    NN_layer( threads ),
    m_mmt_rate{ mmt_rate },
    m_w{ w_sh },
    m_b{ b_sh },
    m_w_mmt{ w_sh, 0 },
    m_b_mmt{ b_sh, 0 },
    m_act_func{ act_func },
    m_err_func{ err_func }
{
    if (nullptr != m_err_func)
    {
        this->m_act_func = this->m_err_func->get_act_func();
    }
}

neurons::Traditional_NN_layer::Traditional_NN_layer(const Traditional_NN_layer & other)
    : 
    NN_layer(other),
    m_mmt_rate(other.m_mmt_rate),
    m_w{ other.m_w }, m_b{ other.m_b },
    m_w_mmt{ other.m_w_mmt },
    m_b_mmt{ other.m_b_mmt },
    m_act_func{ other.m_act_func ? other.m_act_func->clone() : nullptr },
    m_err_func{ other.m_err_func ? other.m_err_func->clone() : nullptr }
{}

neurons::Traditional_NN_layer::Traditional_NN_layer(Traditional_NN_layer && other)
    : 
    NN_layer(other),
    m_mmt_rate(other.m_mmt_rate),
    m_w{ std::move(other.m_w) }, m_b{ std::move(other.m_b) },
    m_w_mmt{ std::move(other.m_w_mmt) },
    m_b_mmt{ std::move(other.m_b_mmt) },
    m_act_func{ std::move(other.m_act_func) },
    m_err_func{ std::move(other.m_err_func) }
{}

neurons::Traditional_NN_layer & neurons::Traditional_NN_layer::operator = (const Traditional_NN_layer & other)
{
    NN_layer::operator=(other);
    this->m_mmt_rate = other.m_mmt_rate;
    this->m_w = other.m_w;
    this->m_b = other.m_b;
    this->m_w_mmt = other.m_w_mmt,
    this->m_b_mmt = other.m_b_mmt,
    this->m_act_func = other.m_act_func ? other.m_act_func->clone() : nullptr;
    this->m_err_func = other.m_err_func ? other.m_err_func->clone() : nullptr;

    return *this;
}


neurons::Traditional_NN_layer & neurons::Traditional_NN_layer::operator = (Traditional_NN_layer && other)
{
    NN_layer::operator=(other);
    this->m_mmt_rate = other.m_mmt_rate;
    this->m_w = std::move(other.m_w);
    this->m_b = std::move(other.m_b);
    this->m_w_mmt = std::move(other.m_w_mmt);
    this->m_b_mmt = std::move(other.m_w_mmt);
    this->m_act_func = std::move(other.m_act_func);
    this->m_err_func = std::move(other.m_err_func);

    return *this;
}

neurons::TMatrix<> neurons::Traditional_NN_layer::weights() const
{
    return this->m_w;
}

neurons::TMatrix<> neurons::Traditional_NN_layer::bias() const
{
    return this->m_b;
}

double neurons::Traditional_NN_layer::commit_training()
{
    TMatrix<> w_gradient_sum{ this->m_w.shape(), 0 };
    TMatrix<> b_gradient_sum{ this->m_b.shape(), 0 };
    double loss = 0;

    for (size_t i = 0; i < this->m_ops.size(); ++i)
    {
        auto op = dynamic_cast<Traditional_NN_layer_op*>(this->m_ops[i].get());

        w_gradient_sum += op->get_weight_gradient();
        b_gradient_sum += op->get_bias_gradient();

        loss += this->m_ops[i]->get_loss();
        this->m_ops[i]->clear_loss();
    }

    this->m_w_mmt = this->m_mmt_rate * this->m_w_mmt + (1 - this->m_mmt_rate) * w_gradient_sum;
    this->m_b_mmt = this->m_mmt_rate * this->m_b_mmt + (1 - this->m_mmt_rate) * b_gradient_sum;

    this->m_w -= this->m_w_mmt;
    this->m_b -= this->m_b_mmt;

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


/*
Structure of a neural network layer:
<
    <size of this layer, 64 bit>

    <layer type len, layer name>

    <weight matrix data>
    <bias matrix data>

    <act function name len, 8 bit><act function name>

    <error function name len, 8 bit><error function name>
>
*/
std::unique_ptr<char[]> neurons::Traditional_NN_layer::to_binary_data(lint & data_size) const
{
    lint w_size, b_size;
    auto w_data = this->m_w.to_binary_data(w_size);
    auto b_data = this->m_b.to_binary_data(b_size);

    std::string nn_type = this->nn_type();

    std::string act_func{ "NULL" };
    std::string err_func{ "NULL" };

    if (this->m_act_func)
    {
        act_func = this->m_act_func->to_string();
    }

    if (this->m_err_func)
    {
        err_func = this->m_err_func->to_string();
    }

    lint size =
        sizeof(uint8_t) + nn_type.size() +
        w_size + b_size +
        sizeof(uint8_t) + act_func.size() + sizeof(uint8_t) + err_func.size();
    data_size = sizeof(lint) + size;

    char * layer_data = new char[data_size];
    lint * size_ptr = reinterpret_cast<lint *>(layer_data);
    *size_ptr = size;

    // Copy name of NN type
    char * position = layer_data + sizeof(lint);
    *(reinterpret_cast<uint8_t *>(position)) = nn_type.size();
    position += sizeof(uint8_t);
    memcpy(position, nn_type.c_str(), nn_type.size());
    
    // Copy w
    position += nn_type.size();
    memcpy(position, w_data.get(), w_size);
    
    // Copy b
    position += w_size;
    memcpy(position, b_data.get(), b_size);
    
    // Copy name of activation function
    position += b_size;
    *(reinterpret_cast<uint8_t *>(position)) = act_func.size();
    position += sizeof(uint8_t);
    memcpy(position, act_func.c_str(), act_func.size());

    // Copy name of error function
    position += act_func.size();
    *(reinterpret_cast<uint8_t *>(position)) = err_func.size();
    position += sizeof(uint8_t);
    memcpy(position, err_func.c_str(), err_func.size());


    return std::unique_ptr<char[]>(layer_data);
}

neurons::Traditional_NN_layer_op::Traditional_NN_layer_op(
    const TMatrix<> & w,
    const TMatrix<> & b,
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

neurons::TMatrix<> neurons::Traditional_NN_layer_op::forward_propagate(const TMatrix<> & input)
{
    std::vector<TMatrix<>> inputs;
    inputs.push_back(input);

    return this->batch_forward_propagate(inputs)[0];
}

neurons::TMatrix<> neurons::Traditional_NN_layer_op::forward_propagate(const TMatrix<> & input, const TMatrix<> & target)
{
    std::vector<TMatrix<>> inputs, targets;
    inputs.push_back(input);
    targets.push_back(target);

    return this->batch_forward_propagate(inputs, targets)[0];
}

neurons::TMatrix<> neurons::Traditional_NN_layer_op::back_propagate(double l_rate, const TMatrix<> & E_to_y_diff)
{
    std::vector<TMatrix<>> E_to_y_diffs;
    E_to_y_diffs.push_back(E_to_y_diff);

    return this->batch_back_propagate(l_rate, E_to_y_diffs)[0];
}

neurons::TMatrix<> neurons::Traditional_NN_layer_op::back_propagate(double l_rate)
{
    return this->batch_back_propagate(l_rate)[0];
}

neurons::TMatrix<>& neurons::Traditional_NN_layer_op::get_weight_gradient() const
{
    return this->m_w_gradient;
}

neurons::TMatrix<>& neurons::Traditional_NN_layer_op::get_bias_gradient() const
{
    return this->m_b_gradient;
}

void neurons::Traditional_NN_layer_op::update_w_and_b(const TMatrix<> & w, const TMatrix<> & b)
{
    this->m_w = w;
    this->m_b = b;
}