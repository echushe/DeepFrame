#include "Conv_NN.h"
#include <fstream>

Conv_NN::Conv_NN(
    double l_rate,
    double mmt_rate,
    lint threads,
    const std::string & model_file,
    const dataset::Dataset &d_set)
    : NN(l_rate, mmt_rate, threads, model_file, d_set)
{
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

void Conv_NN::print_layers(std::ostream & os) const
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

void Conv_NN::save_layers_as_images() const
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

bool Conv_NN::load(const std::string & file_name)
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

    // Always cache shape of the previous layer
    neurons::Shape input_shape{ this->m_train_set[0].shape() };

    while (len_left > 0)
    {
        lint size;
        neurons::TMatrix<> w, b;
        lint stride, padding;
        std::unique_ptr<neurons::Activation> act_func;
        std::unique_ptr<neurons::ErrorFunction> err_func;
        char * re; lint re_len;

        std::string nn_type = 
            neurons::CNN_layer::from_binary_data(
                position, size, w, b, stride, padding, act_func, err_func, re, re_len);
        len_left -= size;
        position += size;

        if ("FCNN" == nn_type)
        {
            this->m_layers.push_back(
                std::make_shared<neurons::FCNN_layer>(this->m_mmt_rate, this->m_threads, w, b, act_func, err_func));

            input_shape = b.shape();
        }
        else if ("CNN" == nn_type)
        {
            /*
            lint rows, lint cols, lint chls, lint stride, lint padding, lint threads,
                TMatrix<>& w, TMatrix<>& b,
                std::unique_ptr<Activation>& act_func, std::unique_ptr<ErrorFunction>& err_func)
            */

            this->m_layers.push_back(
                std::make_shared<neurons::CNN_layer>(
                    this->m_mmt_rate,
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                    stride,
                    padding,
                    this->m_threads,
                    w, b, act_func, err_func));

            input_shape = this->m_layers[this->m_layers.size() - 1]->output_shape();
        }

    }



    return true;
}

bool Conv_NN::load_until(const std::string & file_name, lint layer_index)
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

    // Always cache shape of the previous layer
    neurons::Shape input_shape{ this->m_train_set[0].shape() };

    lint index = 0;
    while (len_left > 0 && index < layer_index)
    {
        lint size;
        neurons::TMatrix<> w, b;
        lint stride, padding;
        std::unique_ptr<neurons::Activation> act_func;
        std::unique_ptr<neurons::ErrorFunction> err_func;
        char * re; lint re_len;

        std::string nn_type = 
            neurons::CNN_layer::from_binary_data(
                position, size, w, b, stride, padding, act_func, err_func, re, re_len);
        len_left -= size;
        position += size;

        if ("FCNN" == nn_type)
        {
            this->m_layers.push_back(
                std::make_shared<neurons::FCNN_layer>(this->m_mmt_rate, this->m_threads, w, b, act_func, err_func));

            input_shape = b.shape();
        }
        else if ("CNN" == nn_type)
        {
            /*
            lint rows, lint cols, lint chls, lint stride, lint padding, lint threads,
                TMatrix<>& w, TMatrix<>& b,
                std::unique_ptr<Activation>& act_func, std::unique_ptr<ErrorFunction>& err_func)
            */

            this->m_layers.push_back(
                std::make_shared<neurons::CNN_layer>(
                    this->m_mmt_rate,
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                    stride,
                    padding,
                    this->m_threads,
                    w, b, act_func, err_func));

            input_shape = this->m_layers[this->m_layers.size() - 1]->output_shape();
        }

        ++index;

    }


    if (this->m_layers.empty())
    {
        this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
            this->m_mmt_rate,
            this->m_train_set[0].shape().size(),
            this->m_train_labels[0].shape().size(),
            this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));
    }
    else
    {
        this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
            this->m_mmt_rate,
            this->m_layers[this->m_layers.size() - 1]->output_shape().size(),
            this->m_train_labels[0].shape().size(),
            this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));
    }


    return true;
}

void Conv_NN::initialize_model()
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
            0, // padding
            this->m_threads,
            new neurons::Tanh));

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
            this->m_mmt_rate,
            this->m_layers[0]->output_shape()[1],
            this->m_layers[0]->output_shape()[2],
            this->m_layers[0]->output_shape()[3],
            10, // filters
            5, // filter rows
            5, // filter cols
            1, // stride
            0, // padding
            this->m_threads,
            new neurons::Tanh));

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
            this->m_mmt_rate,
            this->m_layers[1]->output_shape()[1],
            this->m_layers[1]->output_shape()[2],
            this->m_layers[1]->output_shape()[3],
            20, // filters
            5, // filter rows
            5, // filter cols
            1, // stride
            0, // padding
            this->m_threads,
            new neurons::Tanh));

    this->m_layers.push_back(
        std::make_shared<neurons::CNN_layer>(
            this->m_mmt_rate,
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
            this->m_mmt_rate,
            this->m_layers[3]->output_shape().size(),
            output_size,
            this->m_threads,
            nullptr,
            new neurons::Softmax_CrossEntropy));
}

void Conv_NN::save(const std::string & file_name) const
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


std::vector<neurons::TMatrix<>> Conv_NN::predict(
    const std::vector<neurons::TMatrix<>>& inputs, lint thread_id) const
{
    std::vector<neurons::TMatrix<>> l_inputs = inputs;
    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].left_extend_shape();
        l_inputs[i].normalize();
    }

    for (size_t i = 0; i < this->m_layers.size() - 1; ++i)
    {
        l_inputs = this->m_layers[i]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);
    }

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, l_inputs[i].shape().size() });
    }

    std::vector<neurons::TMatrix<>> preds =
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);

    for (size_t i = 0; i < preds.size(); ++i)
    {
        preds[i].reshape(neurons::Shape{ preds[i].shape().size() });
    }

    return preds;
}


std::vector<neurons::TMatrix<>> Conv_NN::test(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> l_inputs = inputs;

    for (size_t i = 0; i < this->m_layers.size() - 1; ++i)
    {
        l_inputs = this->m_layers[i]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs);
    }

    for (size_t i = 0; i < l_inputs.size(); ++i)
    {
        l_inputs[i].reshape(neurons::Shape{ 1, l_inputs[i].shape().size() });
    }

    std::vector<neurons::TMatrix<>> preds =
        this->m_layers[this->m_layers.size() - 1]->operation_instances()[thread_id]->batch_forward_propagate(l_inputs, targets);

    return preds;
}

std::vector<neurons::TMatrix<>> Conv_NN::optimise(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds = this->test(inputs, targets, thread_id);

    std::vector<neurons::TMatrix<>> E_to_x_diffs =
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




