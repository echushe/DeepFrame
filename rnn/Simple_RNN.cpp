#include "Simple_RNN.h"
#include "Simple_RNN_layer.h"
#include "FCNN_layer.h"

Simple_RNN::Simple_RNN(
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

    lint bptt_len = this->m_train_set[0].shape()[0];
    // Each input sample should have at least 2 dimensions
    lint input_size = this->m_train_set[0].shape()[1];
    
    lint output_size = this->m_train_labels[0].shape().size();

    // Add RNN layer to the network
    this->m_layers.push_back(
        std::make_shared<neurons::Simple_RNN_layer>(input_size, 50, bptt_len, this->m_threads, new neurons::Tanh));

    // Add an output layer
    this->m_layers.push_back(
        std::make_shared<neurons::FCNN_layer>(50, output_size, this->m_threads, nullptr, new neurons::Sigmoid_CrossEntropy));

    // Reshape all the training set and labels
    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        // this->m_train_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_train_set[i].normalize();
        this->m_train_labels[i].reshape(neurons::Shape{ 1, output_size });
    }

    // Reshape all the test set and labels
    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        // this->m_test_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_test_set[i].normalize();
        this->m_test_labels[i].reshape(neurons::Shape{ 1, output_size });
    }
}


std::vector<neurons::Matrix> Simple_RNN::test(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        dynamic_cast<neurons::Simple_RNN_layer_op*>(this->m_layers[0]->operation_instances()[thread_id].get())->forget_all();
        std::vector<neurons::Matrix> sequence = inputs[i].collapse(0);

        neurons::Matrix pred;
        std::vector<neurons::Matrix> s_hiddens;

        // Forward propagate
        for (size_t j = 0; j < sequence.size(); ++j)
        {
            sequence[j].reshape(neurons::Shape{ 1, sequence[j].shape().size() });
        }

        s_hiddens = this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(sequence);

        pred = this->m_layers[1]->operation_instances()[thread_id]->forward_propagate(s_hiddens.back(), targets[i]);

        preds.push_back(pred);
    }

    return preds;
}


std::vector<neurons::Matrix> Simple_RNN::optimise(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        dynamic_cast<neurons::Simple_RNN_layer_op*>(this->m_layers[0]->operation_instances()[thread_id].get())->forget_all();
        std::vector<neurons::Matrix> sequence = inputs[i].collapse(0);
        
        neurons::Matrix pred;
        std::vector<neurons::Matrix> s_hiddens;

        // Forward propagate
        for (size_t j = 0; j < sequence.size(); ++j)
        {
            sequence[j].reshape(neurons::Shape{ 1, sequence[j].shape().size() });
        }
            
        s_hiddens = this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(sequence);

        pred = this->m_layers[1]->operation_instances()[thread_id]->forward_propagate(s_hiddens.back(), targets[i]);

        // Backward propagate
        std::vector<neurons::Matrix> E_to_x_diffs;
        neurons::Matrix E_to_x_diff = this->m_layers[1]->operation_instances()[thread_id]->backward_propagate(this->m_l_rate);
        E_to_x_diffs.push_back(E_to_x_diff);

        this->m_layers[0]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate, E_to_x_diffs);

        preds.push_back(pred);
    }

    return preds;
}


void Simple_RNN::print_layers(std::ostream & os) const
{
    os << '\n';
}

