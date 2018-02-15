#include "Multi_Layer_NN.h"


Multi_Layer_NN::Multi_Layer_NN(
    double l_rate,
    lint batch_size,
    lint threads,
    lint steps,
    lint epoch_size,
    lint secs_allowed,
    const dataset::Dataset &d_set)
    : 
    NN(l_rate, batch_size, threads, steps, epoch_size, secs_allowed, d_set)
{
    // Initialize all layers and
    // reshape all inputs and labels so that they are suitable for matrix multiplications

    lint input_size = this->m_train_set[0].shape().size();
    lint output_size = this->m_train_labels[0].shape().size();

    // Add layers to the network
    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(input_size, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, 100, this->m_threads, new neurons::Relu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(100, output_size, this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));

    // Reshape all the training set and labels
    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_train_set[i].normalize();
        this->m_train_labels[i].reshape(neurons::Shape{ 1, output_size });
    }

    // Reshape all the test set and labels
    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_test_set[i].normalize();
        this->m_test_labels[i].reshape(neurons::Shape{ 1, output_size });
    }
}


std::vector<neurons::Matrix> Multi_Layer_NN::test(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> l_inputs = inputs;
    
    for (size_t i = 0; i < this->m_layers.size() - 1; ++i)
    {
        l_inputs = this->m_layers[i]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);
    }

    std::vector<neurons::Matrix> preds = 
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs, targets);
    return preds;
}


std::vector<neurons::Matrix> Multi_Layer_NN::optimise(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds = this->test(inputs, targets, thread_id);

    std::vector<neurons::Matrix> E_to_x_diffs =
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate);

    for (lint i = this->m_layers.size() - 2; i >= 0; --i)
    {
        E_to_x_diffs = this->m_layers[i]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate, E_to_x_diffs);
    }

    return preds;
}


void Multi_Layer_NN::print_layers(std::ostream & os) const
{
    os << '\n';
}




