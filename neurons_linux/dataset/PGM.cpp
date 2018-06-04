#include "PGM.h"
#include "pgmimage.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <iterator>


std::vector<std::string> dataset::PGM::split(const std::string &s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    
    std::vector<std::string> str_list;
    

    while (std::getline(ss, item, delim))
    {
        str_list.push_back(item);
    }

    return str_list;
}

void dataset::PGM::get_set(
    std::vector<neurons::TMatrix<>>& inputs,
    std::vector<neurons::TMatrix<>>& labels,
    const std::vector<std::vector<std::string>> file_labels,
    const std::vector<std::string> file_names,
    lint limit) const
{
    inputs.clear();
    labels.clear();

    for (const std::string name : file_names)
    {
        IMAGE *image = img_open(name.c_str());
        if (!image)
        {
            std::cout << "Image: " << name << " not found.\n";
            throw std::invalid_argument(std::string("Image file not found"));
        }

        // Get matrix size of this image
        lint mat_size = image->rows * image->cols;
        // Get raw data of this image
        int * pixels = image->data;

        // Malloc binary data of the new matrix
        // This matrix will be 3 dimensional and the same shape as the image
        std::unique_ptr<char[]> binary{ new char[sizeof(lint) * 4 + sizeof(double) * mat_size] };

        // Initialize shape data
        lint * shape_data = reinterpret_cast<lint *>(binary.get());
        shape_data[0] = 3;
        shape_data[1] = image->rows;
        shape_data[2] = image->cols;
        shape_data[3] = 1;

        // Copy raw image data to the matrix
        double * mat_data = reinterpret_cast<double *>(shape_data + 4);
        for (lint i = 0; i < mat_size; ++i)
        {
            mat_data[i] = pixels[i];
        }

        // Add the new matrix to inputs
        lint data_len;
        inputs.push_back(neurons::TMatrix<> {binary.get(), data_len});

        img_free(image);
    }

    for (const auto label_names : file_labels)
    {
        std::string label = label_names[this->m_label_id];
        lint label_index = (*(this->m_label_name_val_map.find(label))).second;

        neurons::TMatrix<> mat_label{ neurons::Shape{ static_cast<lint>(this->m_labels.size()) }, 0 };
        mat_label[{label_index}] = 1;

        labels.push_back(mat_label);
    }
}


dataset::PGM::PGM()
{}

dataset::PGM::PGM(
    const std::vector<std::string>& train_files, const std::vector<std::string>& test_files, lint label_id)
    :
    m_label_id {label_id}
{
    for (const std::string & file_name : train_files)
    {
        // Insert new complete file name
        this->m_train_file_names.push_back(file_name);

        // get the pure file name (removing directory names)
        std::vector<std::string> name_list = this->split(file_name, '/');
        std::string pure_file_name = name_list[name_list.size() - 1];
        
        // Remove the file extension
        name_list = this->split(pure_file_name, '.');
        std::string pure_file_name_without_ext = name_list[0];

        // Split filename into different label types
        name_list = this->split(pure_file_name_without_ext, '_');

        this->m_train_file_labels.push_back(name_list);
        this->m_labels.insert(name_list[label_id]);
    }

    for (const std::string & file_name : test_files)
    {
        // Insert new complete file name
        this->m_test_file_names.push_back(file_name);

        // get the pure file name (removing directory names)
        std::vector<std::string> name_list = this->split(file_name, '/');
        std::string pure_file_name = name_list[name_list.size() - 1];

        // Remove the file extension
        name_list = this->split(pure_file_name, '.');
        std::string pure_file_name_without_ext = name_list[0];

        // Split filename into different label types
        name_list = this->split(pure_file_name_without_ext, '_');

        this->m_test_file_labels.push_back(name_list);
        this->m_labels.insert(name_list[label_id]);
    }

    lint index = 0;
    for (const std::string & label_name : this->m_labels)
    {
        this->m_label_name_val_map[label_name] = index;
        this->m_label_val_name_map[index] = label_name;
        ++index;
    }
}

void dataset::PGM::get_training_set(
    std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    this->get_set(inputs, labels, this->m_train_file_labels, this->m_train_file_names, limit);
}

void dataset::PGM::get_test_set(
    std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    this->get_set(inputs, labels, this->m_test_file_labels, this->m_test_file_names, limit);
}

std::string dataset::PGM::index_to_name(lint pred_index) const
{
    auto itr = this->m_label_val_name_map.find(pred_index);
    if (itr != this->m_label_val_name_map.cend())
    {
        return (*itr).second;
    }
    else
    {
        return std::string{};
    }
}

std::string dataset::PGM::mat_to_name(const neurons::TMatrix<> & pred) const
{
    neurons::TMatrix<> l_pred = pred;
    l_pred.reshape(neurons::Shape{ l_pred.shape().size() });
    neurons::Coordinate argmax = pred.argmax();

    lint pred_index = argmax[0];

    return this->index_to_name(pred_index);
}


