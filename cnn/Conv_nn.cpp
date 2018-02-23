#include "Conv_NN.h"


Conv_NN::Conv_NN(
    double l_rate,
    lint batch_size,
    lint threads,
    lint steps,
    lint epoch_size,
    lint secs_allowed,
    const dataset::Dataset &d_set)
    : NN(l_rate, batch_size, threads, steps, epoch_size, secs_allowed, d_set)
{
    // Initialize all layers and
    // reshape all inputs and labels so that they are suitable for matrix multiplication
    // lint input_size = this->m_train_set[0].shape().size();
    lint output_size = this->m_train_labels[0].shape().size();

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
        this->m_train_set[0].shape()[0],
        this->m_train_set[0].shape()[1],
        this->m_train_set[0].shape()[2],
        6, // filters 
        6, // filter rows
        6, // filter cols
        2, // stride
        0, // padding
        this->m_threads,
        new neurons::Tanh) );

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
        this->m_layers[0]->output_shape()[1],
        this->m_layers[0]->output_shape()[2],
        this->m_layers[0]->output_shape()[3],
        10, // filters
        5, // filter rows
        5, // filter cols
        1, // stride
        0, // padding
        this->m_threads,
        new neurons::Tanh) );

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
        this->m_layers[1]->output_shape()[1],
        this->m_layers[1]->output_shape()[2],
        this->m_layers[1]->output_shape()[3],
        20, // filters
        5, // filter rows
        5, // filter cols
        1, // stride
        0, // padding
        this->m_threads,
        new neurons::Tanh) );

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
        this->m_layers[2]->output_shape()[1],
        this->m_layers[2]->output_shape()[2],
        this->m_layers[2]->output_shape()[3],
        40, // filters
        3, // filter rows
        3, // filter cols
        1, // stride
        0, // padding
        this->m_threads,
        new neurons::Tanh));

    this->m_layers.push_back(
        std::make_shared<neurons::FCNN_layer>(
        this->m_layers[3]->output_shape().size(),
        output_size,
        this->m_threads,
        nullptr,
        new neurons::Softmax_CrossEntropy ));

    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].left_extend_shape();
        this->m_train_set[i].normalize();
        this->m_train_labels[i].reshape(neurons::Shape{ 1, output_size });
    }

    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].left_extend_shape();
        this->m_test_set[i].normalize();
        this->m_test_labels[i].reshape(neurons::Shape{ 1, output_size });
    }
}

void Conv_NN::print_layers(std::ostream & os) const
{
    os << '\n';
}

std::vector<neurons::Matrix> Conv_NN::test(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> l_inputs = inputs;

    for (size_t i = 0; i < this->m_layers.size() - 1; ++i)
    {
        l_inputs = this->m_layers[i]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);
    }

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, l_inputs[i].shape().size() });
    }

    std::vector<neurons::Matrix> preds =
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs, targets);

    return preds;
}

std::vector<neurons::Matrix> Conv_NN::optimise(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds = this->test(inputs, targets, thread_id);

    std::vector<neurons::Matrix> E_to_x_diffs =
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate);

    for (size_t i = 0; i < E_to_x_diffs.size(); ++i)
    {
        E_to_x_diffs[i].reshape(this->m_layers[this->m_layers.size() - 2]->output_shape());
    }

    for (lint i = this->m_layers.size() - 2; i >= 0; --i)
    {
        E_to_x_diffs = this->m_layers[i]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate, E_to_x_diffs);
    }

    return preds;
}




