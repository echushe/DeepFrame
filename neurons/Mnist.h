#pragma once

#include "Matrix.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace mnist
{
    typedef long long int lint;
    /*!
    * \brief Extract the MNIST header from the given buffer
    * \param buffer The current buffer
    * \param position The current reading positoin
    * \return The value of the mnist header
    */
    inline int64_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);

    /*!
    * \brief Read a MNIST file inside a raw buffer
    * \param path The path to the image file
    * \return The buffer of byte on success, a nullptr-unique_ptr otherwise
    */
    inline std::unique_ptr<char[]> read_mnist_file(const std::string & path, uint32_t key);

    void read_mnist_image_file(std::vector<neurons::Matrix> & images, const std::string& path, lint limit = 0);

    void read_mnist_label_file(std::vector<neurons::Matrix> & labels, const std::string& path, lint limit = 0);
}
