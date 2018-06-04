#pragma once
#include "Dataset.h"
#include <string>
#include <vector>
#include <set>
#include <map>


namespace dataset
{
    class PGM : public Dataset
    {
    private:
        lint m_label_id;
        std::vector<std::vector<std::string>> m_train_file_labels;
        std::vector<std::string> m_train_file_names;
        std::vector<std::vector<std::string>> m_test_file_labels;
        std::vector<std::string> m_test_file_names;
        std::set<std::string> m_labels;
        std::map<std::string, lint> m_label_name_val_map;
        std::map<lint, std::string> m_label_val_name_map;

    public:
        PGM();

        PGM(const std::vector<std::string> & train_files, const std::vector<std::string> & test_files, lint label_id);

    public:
        virtual void get_training_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const;

        virtual void get_test_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const;

        std::string index_to_name(lint pred_index) const;

        std::string mat_to_name(const neurons::TMatrix<> & pred) const;

    private:

        std::vector<std::string> split(const std::string &s, char delim);

        void get_set(
            std::vector<neurons::TMatrix<>> & inputs,
            std::vector<neurons::TMatrix<>> & labels,
            const std::vector<std::vector<std::string>> m_file_labels,
            const std::vector<std::string> m_file_names,
            lint limit = 0) const;

    };
}