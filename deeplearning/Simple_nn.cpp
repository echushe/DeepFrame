#include "Simple_nn.h"
#include "Mnist.h"

Simple_nn::Simple_nn(
    double l_rate,
    lint batch_size,
    lint threads,
    lint steps,
    lint epoch_size,
    lint secs_allowed,
    const std::string & train_file,
    const std::string & train_label,
    const std::string & test_file,
    const std::string & test_label)
    : 
    Mnist_nn(l_rate, batch_size, threads, steps, epoch_size, secs_allowed, train_file, train_label, test_file, test_label)
{

    // Initialize the first layer and
    // reshape all inputs and labels so that they are suitable for matrix multiplication

    lint input_size = this->m_train_set[0].shape().size();
    lint output_size = this->m_train_labels[0].shape().size();

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(input_size, output_size, this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));

    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].reshape(neurons::Shape{ 1, input_size });
        this->m_train_set[i].normalize(-1, 1);
        this->m_train_labels[i].reshape(neurons::Shape{ 1, output_size });
    }

    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].reshape(neurons::Shape{ 1, input_size });;
        this->m_test_set[i].normalize(-1, 1);
        this->m_test_labels[i].reshape(neurons::Shape{ 1, output_size });
    }
}

void Simple_nn::print_layers(std::ostream & os) const
{
    os << '\n';
}


std::vector<neurons::Matrix> Simple_nn::foward_propagate(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds =
        this->m_layers[0]->operation_instances()[thread_id]->forward_propagate(inputs, targets);
    return preds;
}


std::vector<neurons::Matrix> Simple_nn::gradient_descent(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds = this->foward_propagate(inputs, targets, thread_id);

    std::vector<neurons::Matrix> E_to_x_diffs =
        this->m_layers[0]->operation_instances()[thread_id]->backward_propagate(this->m_l_rate);

    return preds;
}

