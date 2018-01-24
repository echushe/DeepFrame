#include "Simple_RNN.h"


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

    lint input_size = this->m_train_set[0].shape().size();
    lint output_size = this->m_train_labels[0].shape().size();

    // Add layers to the network
    this->m_layers.push_back(std::make_shared<neurons::Simple_RNN_layer>(input_size, output_size, this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));

    // Reshape all the training set and labels
    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_train_set[i].normalize(-1, 1);
        this->m_train_labels[i].reshape(neurons::Shape{ 1, output_size });
    }

    // Reshape all the test set and labels
    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_test_set[i].normalize(-1, 1);
        this->m_test_labels[i].reshape(neurons::Shape{ 1, output_size });
    }
}


std::vector<neurons::Matrix> Simple_RNN::test(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    /*
    std::vector<neurons::Matrix> preds =
        this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(inputs, targets);
    */

    std::vector<neurons::Matrix> preds;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        preds.push_back(this->m_layers[0]->operation_instances()[thread_id]->forward_propagate(inputs[i], targets[i]));
    }

    return preds;
}


std::vector<neurons::Matrix> Simple_RNN::optimise(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    /*
    std::vector<neurons::Matrix> preds = this->test(inputs, targets, thread_id);

    this->m_layers[0]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate);
    */

    std::vector<neurons::Matrix> preds;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        preds.push_back(this->m_layers[0]->operation_instances()[thread_id]->forward_propagate(inputs[i], targets[i]));
        this->m_layers[0]->operation_instances()[thread_id]->backward_propagate(this->m_l_rate);
    }

    return preds;
}


void Simple_RNN::print_layers(std::ostream & os) const
{
    os << '\n';
}

