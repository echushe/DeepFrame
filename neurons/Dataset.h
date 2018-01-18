#pragma once

#include "Matrix.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace dataset
{
    typedef long long int lint;

    /*
    This is an abstract interface of Dataset to read inputs and labels
    */
    class Dataset
    {
    public:
        virtual void get_training_set(
            std::vector<neurons::Matrix> & inputs, std::vector<neurons::Matrix> & labels, lint limit = 0) const = 0;

        virtual void get_test_set(
            std::vector<neurons::Matrix> & inputs, std::vector<neurons::Matrix> & labels, lint limit = 0) const = 0;
    };


    class Mnist : public Dataset
    {
    private:
        std::string m_train_file;
        std::string m_train_label;
        std::string m_test_file;
        std::string m_test_label;

    private:
        /*!
        * \brief Extract the MNIST header from the given buffer
        * \param buffer The current buffer
        * \param position The current reading positoin
        * \return The value of the mnist header
        */
        int64_t read_header(const std::unique_ptr<char[]>& buffer, size_t position) const;

        /*!
        * \brief Read a MNIST file inside a raw buffer
        * \param path The path to the image file
        * \return The buffer of byte on success, a nullptr-unique_ptr otherwise
        */
        inline std::unique_ptr<char[]> read_mnist_file(const std::string & path, uint32_t key) const;

        void read_mnist_image_file(std::vector<neurons::Matrix> & images, const std::string& path, lint limit = 0) const;

        void read_mnist_label_file(std::vector<neurons::Matrix> & labels, const std::string& path, lint limit = 0) const;

    public:

        Mnist(
            const std::string & train_file,
            const std::string & train_label,
            const std::string & test_file,
            const std::string & test_label);

        virtual void get_training_set(
            std::vector<neurons::Matrix> & inputs, std::vector<neurons::Matrix> & labels, lint limit = 0) const;

        virtual void get_test_set(
            std::vector<neurons::Matrix> & inputs, std::vector<neurons::Matrix> & labels, lint limit = 0) const;
    };
}
