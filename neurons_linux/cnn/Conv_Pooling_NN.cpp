#include "Conv_Pooling_NN.h"


Conv_Pooling_NN::Conv_Pooling_NN(
    double l_rate,
    double mmt_rate,
    lint threads,
    const std::string & model_file,
    const dataset::Dataset &d_set)
    : NN(l_rate, mmt_rate, threads, model_file, d_set)
{
    // Initialize all layers and
    // reshape all inputs and labels so that they are suitable for matrix multiplication
    // lint input_size = this->m_train_set[0].shape().size();
    lint output_size = this->m_train_labels[0].shape().size();

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

    // Initialize all layers
    if (!this->load(this->m_model_file))
    {
        this->initialize_model();
    }
}


void Conv_Pooling_NN::print_layers(std::ostream & os) const
{
    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        neurons::TMatrix<> w = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->weights();
        neurons::TMatrix<> b = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->bias();

        os << "=================== Layer: " << i << " ====================\n";
        os << "Weights:\n";
        os << w;
        os << "Bias:\n";
        os << b;
    }
}

void Conv_Pooling_NN::save_layers_as_images() const
{
    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        neurons::TMatrix<> w = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->weights();
        neurons::TMatrix<> b = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->bias();

        w.normalize(0, 255);
        std::vector<neurons::TMatrix<>> kernel_list = w.collapse(w.shape().dim() - 1);

        for (size_t j = 0; j < kernel_list.size(); ++j)
        {
            std::vector<neurons::TMatrix<>> kernel_ch = kernel_list[j].collapse(kernel_list[j].shape().dim() - 1);

            for (size_t k = 0; k < kernel_ch.size(); ++k)
            {
                kernel_ch[k].save_matrix_as_image(this->m_model_file + "_layer_" +
                    std::to_string(i) + "_kernel_" + std::to_string(j) + "_channel_" + std::to_string(k) + ".pgm");
            }
        }
    }
}

bool Conv_Pooling_NN::load(const std::string & file_name)
{
    return false;
}

bool Conv_Pooling_NN::load_until(const std::string & file_name, lint layer_index)
{
    return false;
}

void Conv_Pooling_NN::initialize_model()
{
    lint output_size = this->m_train_labels[0].shape().size();

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
            this->m_mmt_rate,
            this->m_train_set[0].shape()[1],
            this->m_train_set[0].shape()[2],
            this->m_train_set[0].shape()[3],
            6, // filters 
            6, // filter rows
            6, // filter cols
            2, // stride
            2, // padding
            this->m_threads,
            new neurons::Tanh));

    this->m_pooling_layers.push_back(
        neurons::Pooling_layer{
        this->m_layers[0]->output_shape(),
        neurons::Shape{ 1, 2, 2, this->m_layers[0]->output_shape()[this->m_layers[0]->output_shape().dim() - 1] },
        this->m_threads
    });

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
            this->m_mmt_rate,
            this->m_pooling_layers[0].output_shape()[1],
            this->m_pooling_layers[0].output_shape()[2],
            this->m_pooling_layers[0].output_shape()[3],
            30, // filters
            3, // filter rows
            3, // filter cols
            1, // stride
            1, // padding
            this->m_threads,
            new neurons::Tanh));

    this->m_pooling_layers.push_back(
        neurons::Pooling_layer{
        this->m_layers[1]->output_shape(),
        neurons::Shape{ 1, 2, 2, this->m_layers[1]->output_shape()[this->m_layers[1]->output_shape().dim() - 1] },
        this->m_threads
    });

    this->m_layers.push_back(
        std::make_shared<neurons::FCNN_layer>(
            this->m_mmt_rate,
            this->m_pooling_layers[1].output_shape().size(), output_size, this->m_threads,
            nullptr, new neurons::Softmax_CrossEntropy));
}

void Conv_Pooling_NN::save(const std::string & file_name) const
{
}

std::vector<neurons::TMatrix<>> Conv_Pooling_NN::predict(
    const std::vector<neurons::TMatrix<>>& inputs, lint thread_id) const
{
    std::vector<neurons::TMatrix<>> l_inputs = inputs;
    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].left_extend_shape();
    }

    // std::cout << "============================= forward propagation =============================\n";
    // std::cout << l_inputs[0];

    l_inputs = this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    l_inputs = this->m_pooling_layers[0].operation_instances()[thread_id]->forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    l_inputs = this->m_layers[1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    l_inputs = this->m_pooling_layers[1].operation_instances()[thread_id]->forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, l_inputs[i].shape().size() });
    }

    std::vector<neurons::TMatrix<>> preds =
        this->m_layers[2]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    for (size_t i = 0; i < preds.size(); ++i)
    {
        preds[i].reshape(neurons::Shape{ preds[i].shape().size() });
    }

    return preds;
}


std::vector<neurons::TMatrix<>> Conv_Pooling_NN::test(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> l_inputs = inputs;

    // std::cout << "============================= forward propagation =============================\n";
    // std::cout << l_inputs[0];

    l_inputs = this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    l_inputs = this->m_pooling_layers[0].operation_instances()[thread_id]->forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    l_inputs = this->m_layers[1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    l_inputs = this->m_pooling_layers[1].operation_instances()[thread_id]->forward_propagate(l_inputs);

    // std::cout << l_inputs[0];

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, l_inputs[i].shape().size() });
    }

    std::vector<neurons::TMatrix<>> preds =
        this->m_layers[2]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs, targets);

    return preds;
}


std::vector<neurons::TMatrix<>> Conv_Pooling_NN::optimise(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds = this->test(inputs, targets, thread_id);

    // std::cout << "============================= back propagation =============================\n";

    std::vector<neurons::TMatrix<>> E_to_x_diffs =
        this->m_layers[2]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate);

    for (size_t i = 0; i < E_to_x_diffs.size(); ++i)
    {
        E_to_x_diffs[i].reshape(this->m_pooling_layers[1].output_shape());
    }

    // std::cout << E_to_x_diffs[0];

    E_to_x_diffs = this->m_pooling_layers[1].operation_instances()[thread_id]->back_propagate(E_to_x_diffs);

    // std::cout << E_to_x_diffs[0];

    E_to_x_diffs = this->m_layers[1]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate, E_to_x_diffs);

    // std::cout << E_to_x_diffs[0];

    E_to_x_diffs = this->m_pooling_layers[0].operation_instances()[thread_id]->back_propagate(E_to_x_diffs);

    // std::cout << E_to_x_diffs[0];

    E_to_x_diffs = this->m_layers[0]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate, E_to_x_diffs);

    return preds;
}






