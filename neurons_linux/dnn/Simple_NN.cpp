#include "Simple_NN.h"
#include <fstream>

Simple_NN::Simple_NN(
    double l_rate,
    double mmt_rate,
    lint threads,
    const std::string & model_file,
    const dataset::Dataset &d_set)
    : 
    NN(l_rate, mmt_rate, threads, model_file, d_set),
    m_input_size{ m_train_set[0].shape().size() },
    m_output_size{ m_train_labels[0].shape().size() }
{
    // Initialize all layers
    if (!this->load(this->m_model_file))
    {
        this->initialize_model();
    }

    for (size_t i = 0; i < this->m_train_set.size(); ++i)
    {
        this->m_train_set[i].reshape(neurons::Shape{ 1, m_input_size });
        this->m_train_set[i].normalize();
        this->m_train_labels[i].reshape(neurons::Shape{ 1, m_output_size });
    }

    for (size_t i = 0; i < this->m_test_set.size(); ++i)
    {
        this->m_test_set[i].reshape(neurons::Shape{ 1, m_input_size });;
        this->m_test_set[i].normalize();
        this->m_test_labels[i].reshape(neurons::Shape{ 1, m_output_size });
    }
}

void Simple_NN::print_layers(std::ostream & os) const
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

void Simple_NN::save_layers_as_images() const
{
    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        neurons::TMatrix<> w = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->weights();
        neurons::TMatrix<> b = (dynamic_cast<neurons::Traditional_NN_layer *>(this->m_layers[i].get()))->bias();

        w.normalize(0, 255);
        w.save_matrix_as_image(this->m_model_file + "_layer_" + std::to_string(i) + ".pgm");
    }
}

bool Simple_NN::load(const std::string & file_name)
{
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

    char * position = buffer.get();

    lint size;
    neurons::TMatrix<> w, b;
    std::unique_ptr<neurons::Activation> act_func;
    std::unique_ptr<neurons::ErrorFunction> err_func;
    char * re; lint re_len;

    neurons::Traditional_NN_layer::from_binary_data(position, size, w, b, act_func, err_func, re, re_len);


    this->m_layers.push_back(
        std::make_shared<neurons::FCNN_layer>(this->m_mmt_rate, this->m_threads, w, b, act_func, err_func));

    return true;
}

bool Simple_NN::load_until(const std::string & file_name, lint layer_index)
{
    return false;
}

void Simple_NN::initialize_model()
{
    // Initialize the first layer
    this->m_layers.push_back(std::make_shared<neurons::FCNN_layer>(
        this->m_mmt_rate,
        m_input_size, m_output_size, this->m_threads, nullptr, new neurons::Softmax_CrossEntropy));
}


void Simple_NN::save(const std::string & file_name) const
{
    lint data_size = 0;

    std::ofstream out_file;
    out_file.open(file_name, std::ios::out | std::ios::binary);

    lint size;
    std::unique_ptr<char[]> data = this->m_layers[0]->to_binary_data(size);

    out_file.write(data.get(), size);

    out_file.close();
}


std::vector<neurons::TMatrix<>> Simple_NN::predict(
    const std::vector<neurons::TMatrix<>>& inputs, lint thread_id) const
{
    std::vector<neurons::TMatrix<>> preds =
        this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(inputs);

    for (size_t i = 0; i < preds.size(); ++i)
    {
        preds[i].reshape(neurons::Shape{ preds[i].shape().size() });
    }

    return preds;
}


std::vector<neurons::TMatrix<>> Simple_NN::test(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds =
        this->m_layers[0]->operation_instances()[thread_id]->batch_forward_propagate(inputs, targets);
    return preds;
}


std::vector<neurons::TMatrix<>> Simple_NN::optimise(
    const std::vector<neurons::TMatrix<>>& inputs,
    const std::vector<neurons::TMatrix<>>& targets,
    lint thread_id)
{
    std::vector<neurons::TMatrix<>> preds = this->test(inputs, targets, thread_id);

    std::vector<neurons::TMatrix<>> E_to_x_diffs =
        this->m_layers[0]->operation_instances()[thread_id]->batch_back_propagate(this->m_l_rate);

    return preds;
}

