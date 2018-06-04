#include "Simple_RNN.h"
#include "Simple_RNN_layer.h"
#include "FCNN_layer.h"

Simple_RNN::Simple_RNN(
    double l_rate,
    double mmt_rate,
    lint threads,
    const std::string & model_file,
    const dataset::Dataset &d_set)
    :
    NN(l_rate, mmt_rate, threads, model_file, d_set)
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
        std::make_shared<neurons::FCNN_layer>(0, 50, output_size, this->m_threads, nullptr, new neurons::Sigmoid_CrossEntropy));

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


std::vector<neurons::TMatrix<>> Simple_RNN::test(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        dynamic_cast<neurons::Simple_RNN_layer_op*>(this->m_layers[0]->operation_instances()[thread_id].get())->forget_all();
        std::vector<neurons::TMatrix<>> sequence = inputs[i].collapse(0);

        neurons::TMatrix<> pred;
        std::vector<neurons::TMatrix<>> s_hiddens;

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


std::vector<neurons::TMatrix<>> Simple_RNN::optimise(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        dynamic_cast<neurons::Simple_RNN_layer_op*>(this->m_layers[0]->operation_instances()[thread_id].get())->forget_all();
        std::vector<neurons::TMatrix<>> sequence = inputs[i].collapse(0);
        
        neurons::TMatrix<> pred;
        std::vector<neurons::TMatrix<>> s_hiddens;

        // Forward propagate
        for (size_t j = 0; j < sequence.size(); ++j)
        {
            sequence[j].reshape(neurons::Shape{ 1, sequence[j].shape().size() });
        }
            
        s_hiddens = this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(sequence);

        pred = this->m_layers[1]->operation_instances()[thread_id]->forward_propagate(s_hiddens.back(), targets[i]);

        // Backward propagate
        std::vector<neurons::TMatrix<>> E_to_x_diffs;
        neurons::TMatrix<> E_to_x_diff = this->m_layers[1]->operation_instances()[thread_id]->back_propagate(this->m_l_rate);
        E_to_x_diffs.push_back(E_to_x_diff);

        this->m_layers[0]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate, E_to_x_diffs);

        preds.push_back(pred);
    }

    return preds;
}


void Simple_RNN::print_layers(std::ostream & os) const
{
    os << '\n';
}

void Simple_RNN::save_layers_as_images() const
{
}

bool Simple_RNN::load(const std::string & file_name)
{
    return false;
}

bool Simple_RNN::load_until(const std::string & file_name, lint layer_index)
{
    return false;
}

void Simple_RNN::initialize_model()
{
}

void Simple_RNN::save(const std::string & file_name) const
{
}

std::vector<neurons::TMatrix<>> Simple_RNN::predict(
    const std::vector<neurons::TMatrix<>>& inputs, lint thread_id) const
{
    return std::vector<neurons::TMatrix<>>();
}

