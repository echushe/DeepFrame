#include "Multi_Layer_NN.h"
#include <fstream>

Multi_Layer_NN::Multi_Layer_NN(
    double l_rate,
    double mmt_rate,
    lint threads,
    const std::string & model_file,
    const dataset::Dataset &d_set)
    : 
    NN(l_rate, mmt_rate, threads, model_file, d_set),
    m_input_size{m_train_set[0].shape().size()},
    m_output_size{m_train_labels[0].shape().size()}
{
    // Initialize all layers
    if (!this->load(this->m_model_file))
    {
        this->initialize_model();
    }

    // Reshape all the training set and labels
    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].reshape(neurons::Shape{ 1, m_input_size });
        this->m_train_set[i].normalize();
        this->m_train_labels[i].reshape(neurons::Shape{ 1, m_output_size });
    }

    // Reshape all the test set and labels
    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].reshape(neurons::Shape{ 1, m_input_size });
        this->m_test_set[i].normalize();
        this->m_test_labels[i].reshape(neurons::Shape{ 1, m_output_size });
    }
}


std::vector<neurons::TMatrix<>> Multi_Layer_NN::predict(
    const std::vector<neurons::TMatrix<>>& inputs, lint thread_id) const
{
    std::vector<neurons::TMatrix<>> l_inputs = inputs;
    // Reshape all the input
    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, this->m_input_size });
        l_inputs[i].normalize();
    }

    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        l_inputs = this->m_layers[i]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);
    }

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ l_inputs[i].shape().size() });
    }

    return l_inputs;
}


std::vector<neurons::TMatrix<>> Multi_Layer_NN::test(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> l_inputs = inputs;
    
    for (size_t i = 0; i < this->m_layers.size() - 1; ++i)
    {
        l_inputs = this->m_layers[i]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);
    }

    std::vector<neurons::TMatrix<>> preds = 
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs, targets);
    return preds;
}


std::vector<neurons::TMatrix<>> Multi_Layer_NN::optimise(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds = this->test(inputs, targets, thread_id);

    std::vector<neurons::TMatrix<>> E_to_x_diffs =
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate);

    for (lint i = this->m_layers.size() - 2; i >= 0; --i)
    {
        E_to_x_diffs = this->m_layers[i]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate, E_to_x_diffs);
    }

    return preds;
}


void Multi_Layer_NN::print_layers(std::ostream & os) const
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

void Multi_Layer_NN::save_layers_as_images() const
{
    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        neurons::TMatrix<> w = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->weights();
        neurons::TMatrix<> b = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->bias();

        w.normalize(0, 255);
        w.save_matrix_as_image(this->m_model_file + "_layer_" + std::to_string(i) + ".pgm");
    }
}

bool Multi_Layer_NN::load(const std::string & file_name)
{
    //Do not forget to clear all layers of this network
    this->m_layers.clear();

    lint data_size = 0;

    std::ifstream in_file;
    in_file.open(file_name, std::ios::in | std::ios::binary | std::ios::ate);

    if (!in_file)
    {
        return false;
    }

    lint file_len = in_file.tellg();
    std::unique_ptr<char[]> buffer(new char[file_len]);
    //Read the entire file at once
    in_file.seekg(0, std::ios::beg);
    in_file.read(buffer.get(), file_len);
    in_file.close();

    lint len_left = file_len;
    char * position = buffer.get();

    while (len_left > 0)
    {
        lint size;
        neurons::TMatrix<> w, b;
        std::unique_ptr<neurons::Activation> act_func;
        std::unique_ptr<neurons::ErrorFunction> err_func;
        char * re; lint re_len;

        neurons::Traditional_NN_layer::from_binary_data(position, size, w, b, act_func, err_func, re, re_len);
        len_left -= size;
        position += size;

        this->m_layers.push_back(
            std::make_shared<neurons::FCNN_layer>(this->m_mmt_rate, this->m_threads, w, b, act_func, err_func));
    }

    return true;
}

bool Multi_Layer_NN::load_until(const std::string & file_name, lint layer_index)
{
    //Do not forget to clear all layers of this network
    this->m_layers.clear();

    lint data_size = 0;

    std::ifstream in_file;
    in_file.open(file_name, std::ios::in | std::ios::binary | std::ios::ate);

    if (!in_file)
    {
        return false;
    }

    lint file_len = in_file.tellg();
    std::unique_ptr<char[]> buffer(new char[file_len]);
    //Read the entire file at once
    in_file.seekg(0, std::ios::beg);
    in_file.read(buffer.get(), file_len);
    in_file.close();

    lint len_left = file_len;
    char * position = buffer.get();

    lint index = 0;
    while (len_left > 0 && index < layer_index)
    {
        lint size;
        neurons::TMatrix<> w, b;
        std::unique_ptr<neurons::Activation> act_func;
        std::unique_ptr<neurons::ErrorFunction> err_func;
        char * re; lint re_len;

        neurons::Traditional_NN_layer::from_binary_data(position, size, w, b, act_func, err_func, re, re_len);
        len_left -= size;
        position += size;

        this->m_layers.push_back(
            std::make_shared<neurons::FCNN_layer>(this->m_mmt_rate, this->m_threads, w, b, act_func, err_func));

        ++index;
    }

    if (this->m_layers.empty())
    {
        this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
            this->m_mmt_rate,
            this->m_input_size,
            m_output_size,
            this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));
    }
    else
    {
        this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
            this->m_mmt_rate,
            this->m_layers[this->m_layers.size() - 1]->output_shape().size(),
            m_output_size,
            this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));
    }

    return true;
}

void Multi_Layer_NN::save(const std::string & file_name) const
{
    lint data_size = 0;

    std::ofstream out_file;
    out_file.open(file_name, std::ios::out | std::ios::binary);

    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        lint size;
        std::unique_ptr<char[]> data = this->m_layers[i]->to_binary_data(size);

        out_file.write(data.get(), size);
    }

    out_file.close();
}


void Multi_Layer_NN::initialize_model()
{
    // Add layers to the network
    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
        this->m_mmt_rate, m_input_size, 100, this->m_threads, new neurons::Arctan));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
        this->m_mmt_rate, 100, 100, this->m_threads, new neurons::LeakyRelu));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
        this->m_mmt_rate, 100, 100, this->m_threads, new neurons::Sin));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
        this->m_mmt_rate, 100, 100, this->m_threads, new neurons::Tanh));

    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
        this->m_mmt_rate, 100, m_output_size, this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));

    // this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
    // this->m_mmt_rate, 100, m_output_size, this->m_threads, nullptr, new neurons::HalfSquareError(new neurons::Relu)));
}




