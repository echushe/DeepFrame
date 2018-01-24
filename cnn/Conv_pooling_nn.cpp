#include "Conv_Pooling_NN.h"


Conv_Pooling_NN::Conv_Pooling_NN(
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
    lint chls = 1;

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
        this->m_train_set[0].shape()[0],
        this->m_train_set[0].shape()[1],
        chls,
        2, // filters 
        5, // filter rows
        5, // filter cols
        1, // stride
        2, // padding
        this->m_threads,
        new neurons::Tanh ));

    this->m_pooling_layers.push_back(
        neurons::Pooling_layer{
        this->m_layers[0]->output_shape(),
        neurons::Shape{ 1, 2, 2, this->m_layers[0]->output_shape()[this->m_layers[0]->output_shape().dim() - 1] },
        this->m_threads
    });

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
        this->m_pooling_layers[0].output_shape()[1],
        this->m_pooling_layers[0].output_shape()[2],
        this->m_pooling_layers[0].output_shape()[3],
        20, // filters
        5, // filter rows
        5, // filter cols
        1, // stride
        0, // padding
        this->m_threads,
        new neurons::Tanh ));

    this->m_pooling_layers.push_back(
        neurons::Pooling_layer{
        this->m_layers[1]->output_shape(),
        neurons::Shape{1, 2, 2, this->m_layers[1]->output_shape()[this->m_layers[1]->output_shape().dim() - 1]},
        this->m_threads
    });

    this->m_layers.push_back(
        std::make_shared<neurons::FCNN_layer>(
        this->m_pooling_layers[1].output_shape().size(), output_size, this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));

    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].left_extend_shape().right_extend_shape();
        this->m_train_set[i].normalize(-1, 1);
        this->m_train_labels[i].reshape(neurons::Shape{ 1, output_size });
    }

    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].left_extend_shape().right_extend_shape();
        this->m_test_set[i].normalize(-1, 1);
        this->m_test_labels[i].reshape(neurons::Shape{ 1, output_size });
    }
}


void Conv_Pooling_NN::print_layers(std::ostream & os) const
{
    os << '\n';
}


std::vector<neurons::Matrix> Conv_Pooling_NN::test(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> l_inputs = inputs;

    l_inputs = this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    l_inputs = this->m_pooling_layers[0].operation_instances()[thread_id]->forward_propagate(l_inputs);

    l_inputs = this->m_layers[1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    l_inputs = this->m_pooling_layers[1].operation_instances()[thread_id]->forward_propagate(l_inputs);

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, l_inputs[i].shape().size() });
    }

    std::vector<neurons::Matrix> preds =
        this->m_layers[2]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs, targets);

    return preds;
}


std::vector<neurons::Matrix> Conv_Pooling_NN::optimise(
    const std::vector<neurons::Matrix>& inputs,
    const std::vector<neurons::Matrix>& targets,
    lint thread_id)
{
    std::vector<neurons::Matrix> preds = this->test(inputs, targets, thread_id);

    std::vector<neurons::Matrix> E_to_x_diffs =
        this->m_layers[2]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate);

    for (size_t i = 0; i < E_to_x_diffs.size(); ++i)
    {
        E_to_x_diffs[i].reshape(this->m_pooling_layers[1].output_shape());
    }

    E_to_x_diffs = this->m_pooling_layers[1].operation_instances()[thread_id]->backward_propagate(E_to_x_diffs);

    E_to_x_diffs = this->m_layers[1]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate, E_to_x_diffs);

    E_to_x_diffs = this->m_pooling_layers[0].operation_instances()[thread_id]->backward_propagate(E_to_x_diffs);

    E_to_x_diffs = this->m_layers[0]->operation_instances()[thread_id]->batch_backward_propagate(this->m_l_rate, E_to_x_diffs);

    return preds;
}






