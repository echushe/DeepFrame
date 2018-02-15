#pragma once
#include "Dataset.h"
#include <string>
#include <vector>

namespace dataset
{
    class CIFAR_10 : public Dataset
    {
    private:
        const lint m_image_len = 3072;
        const lint m_label_len = 1;
        const lint m_image_rows = 32;
        const lint m_image_cols = 32;
        const lint m_image_chls = 3;
        const lint m_label_size = 10;

        std::string m_dir;

    private:
        std::unique_ptr<char[]> read_cifar_file(lint & file_len, const std::string & path) const;

        void cifar_binary_to_dataset(
            std::vector<neurons::Matrix> & images,
            std::vector<neurons::Matrix> & labels,
            const std::unique_ptr<char[]> & binary,
            lint binary_len, lint limit = 0) const;

    public:
        CIFAR_10(const std::string & dir);

    public:
        virtual void get_training_set(
            std::vector<neurons::Matrix> & inputs, std::vector<neurons::Matrix> & labels, lint limit = 0) const;

        virtual void get_test_set(
            std::vector<neurons::Matrix> & inputs, std::vector<neurons::Matrix> & labels, lint limit = 0) const;
    };
}

