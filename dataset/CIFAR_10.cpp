#include "CIFAR_10.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <memory>

std::unique_ptr<char[]> dataset::CIFAR_10::read_cifar_file(lint & file_len, const std::string & path) const
{
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file)
    {
        std::cout << "Error opening file" << std::endl;
        return {};
    }

    file_len = file.tellg();
    if (file_len % (this->m_image_len + this->m_label_len))
    {
        std::cout << "cifar file format is wrong" << std::endl;
        return {};
    }

    std::unique_ptr<char[]> buffer(new char[file_len]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_len);
    file.close();

    return buffer;
}

void dataset::CIFAR_10::cifar_binary_to_dataset(
    std::vector<neurons::Matrix>& images,
    std::vector<neurons::Matrix>& labels,
    const std::unique_ptr<char[]>& binary,
    lint binary_len, lint limit) const
{
    uint8_t* u_binary = reinterpret_cast<uint8_t*>(binary.get());
    lint len = binary_len / (this->m_image_len + this->m_label_len);
    lint img_size = this->m_image_rows * this->m_image_cols;

    if (0 == limit)
    {
        limit = len;
    }

    if (len > limit)
    {
        len = limit;
    }

    uint8_t * image_pos = u_binary;
    for (lint i = 0; i < len; ++i)
    {
        neurons::Matrix image{ neurons::Shape{this->m_image_rows, this->m_image_cols, this->m_image_chls} };
        neurons::Matrix label{ neurons::Shape{this->m_label_size}, 0 };
        neurons::Coordinate pixel_pos{ 0, 0, 0 };

        label[{*image_pos}] = 1;
        ++image_pos;

        for (lint k = 0; k < this->m_image_chls; ++k)
        {
            for (lint i = 0; i < this->m_image_rows; ++i)
            {
                for (lint j = 0; j < this->m_image_cols; ++j)
                {
                    pixel_pos[0] = i; pixel_pos[1] = j; pixel_pos[2] = k;
                    image[pixel_pos] = *image_pos;
                    ++image_pos;
                }
            }
        }

        images.push_back(image);
        labels.push_back(label);
    }
}

dataset::CIFAR_10::CIFAR_10(const std::string & dir)
    : m_dir{ dir }
{}

void dataset::CIFAR_10::get_training_set(
    std::vector<neurons::Matrix>& inputs, std::vector<neurons::Matrix>& labels, lint limit) const
{
    limit /= 5;
    lint file_len;

    for (lint i = 0; i < 5; ++i)
    {
        auto buffer = this->read_cifar_file(file_len, this->m_dir + "data_batch_" + std::to_string(i + 1) +".bin");

        if (!buffer)
        {
            throw std::invalid_argument(std::string{ "Error while opening CIFAR file ..." });
        }

        this->cifar_binary_to_dataset(inputs, labels, buffer, file_len, limit);
    }
}

void dataset::CIFAR_10::get_test_set(
    std::vector<neurons::Matrix>& inputs, std::vector<neurons::Matrix>& labels, lint limit) const
{
    lint file_len;

    auto buffer = this->read_cifar_file(file_len, this->m_dir + "test_batch.bin");

    if (!buffer)
    {
        throw std::invalid_argument(std::string{ "Error while opening CIFAR file ..." });
    }

    this->cifar_binary_to_dataset(inputs, labels, buffer, file_len, limit);
}


