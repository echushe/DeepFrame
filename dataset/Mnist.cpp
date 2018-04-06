#include "Mnist.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <memory>

/*!
* \brief Extract the MNIST header from the given buffer
* \param buffer The current buffer
* \param position The current reading positoin
* \return The value of the mnist header
*/
int64_t dataset::Mnist::read_header(const std::unique_ptr<char[]>& buffer, size_t position) const
{
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    auto decode = (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);

    return static_cast<lint>(decode);
}

/*!
* \brief Read a MNIST file inside a raw buffer
* \param path The path to the image file
* \return The buffer of byte on success, a nullptr-unique_ptr otherwise
*/
inline std::unique_ptr<char[]> dataset::Mnist::read_mnist_file(const std::string & path, uint32_t key) const
{
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file)
    {
        std::cout << "Error opening file" << std::endl;
        return {};
    }

    auto size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), size);
    file.close();

    auto magic = read_header(buffer, 0);

    // std::cout << key << std::endl;
    // std::cout << magic << std::endl;

    if (magic != key)
    {
        std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        return {};
    }

    auto count = this->read_header(buffer, 1);

    if (magic == 0x803)
    {
        auto rows = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        if (size < count * rows * columns + 16) {
            std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            return {};
        }
    }
    else if (magic == 0x801)
    {
        if (size < count + 8)
        {
            std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            return {};
        }
    }

    return buffer;
}


void dataset::Mnist::read_mnist_image_file(std::vector<neurons::TMatrix<>> & images, const std::string& path, lint limit) const
{
    auto buffer = this->read_mnist_file(path, 0x803);

    if (buffer)
    {
        auto count = this->read_header(buffer, 1);
        auto rows = this->read_header(buffer, 2);
        auto columns = this->read_header(buffer, 3);

        if (limit > 0 && count > limit)
        {
            count = limit;
        }

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        uint8_t* image_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 16);

        for (lint i = 0; i < count; ++i)
        {
            neurons::TMatrix<> new_image{ neurons::Shape{ rows, columns, 1 } };
            neurons::Coordinate pos{ 0, 0, 0 };
            for (lint j = 0; j < rows; ++j)
            {
                for (lint k = 0; k < columns; ++k)
                {
                    uint8_t pixel = *image_buffer++;
                    pos[0] = j; pos[1] = k;
                    new_image[pos] = pixel;
                }
            }

            images.push_back(new_image);
        }
    }
}


void dataset::Mnist::read_mnist_label_file(std::vector<neurons::TMatrix<>> & labels, const std::string& path, lint limit) const
{
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer)
    {
        auto count = this->read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        uint8_t* label_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 8);

        if (limit > 0 && count > limit)
        {
            count = limit;
        }

        uint8_t max_val = 0;
        for (lint i = 0; i < count; ++i)
        {
            uint8_t label_val = *(label_buffer + i);
            if (label_val > max_val)
            {
                max_val = label_val;
            }
        }

        for (lint i = 0; i < count; ++i)
        {
            neurons::TMatrix<> label{ neurons::Shape{ max_val + 1 } };
            uint8_t label_val = *(label_buffer + i);

            for (uint8_t j = 0; j <= max_val; ++j)
            {
                if (j == label_val)
                {
                    label[{j}] = 1;
                }
                else
                {
                    label[{j}] = 0;
                }
            }

            labels.push_back(label);
        }
    }
}


dataset::Mnist::Mnist(
    const std::string & train_file,
    const std::string & train_label,
    const std::string & test_file,
    const std::string & test_label)
    :
    m_train_file{ train_file },
    m_train_label{ train_label },
    m_test_file{ test_file },
    m_test_label{ test_label }
{}


void dataset::Mnist::get_training_set(
    std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    this->read_mnist_image_file(inputs, this->m_train_file, limit);
    this->read_mnist_label_file(labels, this->m_train_label, limit);
}


void dataset::Mnist::get_test_set(
    std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    this->read_mnist_image_file(inputs, this->m_test_file, limit);
    this->read_mnist_label_file(labels, this->m_test_label, limit);
}
