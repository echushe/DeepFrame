#pragma once
#include "Dataset.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <random>

namespace dataset
{
    class Review : public Dataset
    {

    private:
        std::string m_word_vec_file;
        std::string m_review_dir;

        std::unordered_map<std::string, neurons::TMatrix<>> m_word_vec_dic;
        std::vector<std::vector<std::string>> m_pos_reviews;
        std::vector<std::vector<std::string>> m_neg_reviews;

        lint m_wordvec_len;

        double m_test_rate;

        std::vector<neurons::TMatrix<>> m_inputs;
        std::vector<neurons::TMatrix<>> m_labels;

        size_t m_arbitrary_review_len;

    private:

        void read_word_vec_dic(std::unordered_map<std::string, neurons::TMatrix<>> &dic, const std::string & word_vec_file);

        std::vector<std::string> read_review(const std::string & review_file);

        void read_reviews(std::vector<std::vector<std::string>> & reviews, const std::string & review_dir);

        neurons::TMatrix<> review_to_matrix(const std::vector<std::string> & review) const;

        void get_data_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const;

    public:

        Review(const std::string & word_vec_file, const std::string & review_dir, double test_rate, lint arbitrary_review_len = 0);

        virtual void get_training_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const;

        virtual void get_test_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const;
    };
}

