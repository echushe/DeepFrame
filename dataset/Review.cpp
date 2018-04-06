#include "Review.h"
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>


dataset::Review::Review(
    const std::string & word_vec_file,
    const std::string & review_dir,
    double test_rate,
    lint arbitrary_review_len)
    :
    m_word_vec_file{ word_vec_file },
    m_review_dir{ review_dir },
    m_wordvec_len{ 0 },
    m_test_rate{test_rate},
    m_arbitrary_review_len{static_cast<size_t>(arbitrary_review_len)}
{
    this->read_word_vec_dic(this->m_word_vec_dic, this->m_word_vec_file);
    this->read_reviews(this->m_pos_reviews, this->m_review_dir + "/pos");
    this->read_reviews(this->m_neg_reviews, this->m_review_dir + "/neg");
    this->get_data_set(this->m_inputs, this->m_labels);
}

void dataset::Review::read_word_vec_dic(
    std::unordered_map<std::string, neurons::TMatrix<>> &dic, const std::string & word_vec_file)
{
    std::fstream infile;
    infile.open(word_vec_file);

    if (!infile)
    {
        throw std::invalid_argument(std::string("dataset::Review::read_word_vec_dic: Invalid word to vec dictionary file!"));
    }

    std::stringstream buffer;
    buffer << infile.rdbuf();

    std::string line;

    lint index = 0;
    // Read each line from the word to vec file
    while (std::getline(buffer, line))
    {
        // Try to split a line into separate words via spaces
        std::istringstream buff{ line };
        std::istream_iterator<std::string> begin{ buff }, end;
        std::vector<std::string> tokens(begin, end);

        if (0 == this->m_wordvec_len)
        {
            this->m_wordvec_len = tokens.size() - 1;
        }
        else if (this->m_wordvec_len != tokens.size() - 1)
        {
            throw std::invalid_argument(
                std::string("dataset::Review::read_word_vec_dic: size of vector for each word should be same!"));
        }

        // The first word is key of the dictionary
        std::string key{ tokens[0] };

        // The other words of this line are all considered as float values
        std::vector<double> float_vals;
        
        // Insert these floats in to a vector
        for (lint i = 0; i < m_wordvec_len; ++i)
        {
            float_vals.push_back(std::stod(tokens[i + 1]));
        }

        // Convert the vector of floats into a neurons::Vector
        neurons::Vector vec{ float_vals.begin(), float_vals.end() };

        // Convert the neurons::vector into a neurons::TMatrix<>, and
        // insert it into the word-vec dictionary via the key
        dic.insert(std::make_pair(key, neurons::TMatrix<>{ vec, true }));

        if (0 == index % 10000)
        {
            std::cout << index << '\t' << key << "\n"; // " => " << neurons::TMatrix<>{ vec, true } << "\n";
        }
        ++index;
    }
}


std::vector<std::string> dataset::Review::read_review(const std::string & review_file)
{
    std::fstream infile;
    infile.open(review_file);

    if (!infile)
    {
        throw std::invalid_argument(std::string("dataset::Review::read_review: Invalid file!"));
    }

    std::stringstream buffer;
    buffer << infile.rdbuf();

    std::istream_iterator<std::string> begin{ buffer }, end;
    std::vector<std::string> tokens{ begin, end };

    if (this->m_arbitrary_review_len)
    {
        std::vector<std::string> _tokens;
        size_t review_size = tokens.size();
        for (size_t i = 0; i < this->m_arbitrary_review_len; ++i)
        {
            if (i >= review_size)
            {
                _tokens.push_back("UNK");
            }
            else
            {
                _tokens.push_back(tokens[i]);
            }
        }

        return _tokens;
    }
    else
    {
        return tokens;
    }
}


neurons::TMatrix<> dataset::Review::review_to_matrix(const std::vector<std::string>& review) const
{
    std::vector<neurons::TMatrix<>> review_words_vec;

    for (std::string word : review)
    {
        // Convert this string to a lower case string
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        if (!this->m_word_vec_dic.count(word))
        {
            review_words_vec.push_back(neurons::TMatrix<>{ neurons::Shape{ 1, this->m_wordvec_len }, 0 });
        }
        else
        {
            review_words_vec.push_back(this->m_word_vec_dic.find(word)->second);
        }
    }

    neurons::TMatrix<> review_mat{ review_words_vec };

    review_mat.reshape(neurons::Shape{ static_cast<lint>(review_words_vec.size()), this->m_wordvec_len });

    return review_mat;
}


void dataset::Review::get_data_set(std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    std::default_random_engine label_rand{ 120 };
    std::uniform_int_distribution<int> label_distribution{ 0, 1 };

    std::vector<std::string> label2dir;

    size_t pos_index = 0;
    size_t neg_index = 0;

    size_t all_samples = this->m_pos_reviews.size() + this->m_neg_reviews.size();

    for (size_t i = 0; i < all_samples; ++i)
    {
        neurons::TMatrix<> review_mat;
        neurons::TMatrix<> review_label{ neurons::Shape{ 2 }, 0 };

        if (pos_index == this->m_pos_reviews.size() && neg_index < this->m_neg_reviews.size())
        {
            review_mat = this->review_to_matrix(this->m_pos_reviews[neg_index]);

            review_label[{1}] = 1;

            inputs.push_back(review_mat);
            labels.push_back(review_label);

            ++neg_index;
        }
        else if (neg_index == this->m_neg_reviews.size() && pos_index < this->m_pos_reviews.size())
        {
            review_mat = this->review_to_matrix(this->m_pos_reviews[pos_index]);

            review_label[{0}] = 1;

            inputs.push_back(review_mat);
            labels.push_back(review_label);

            ++pos_index;
        }
        else
        {
            size_t label_id = label_distribution(label_rand);

            if (label_id)
            {
                review_mat = this->review_to_matrix(this->m_pos_reviews[pos_index]);

                review_label[{0}] = 1;

                inputs.push_back(review_mat);
                labels.push_back(review_label);

                ++pos_index;
            }
            else
            {
                review_mat = this->review_to_matrix(this->m_neg_reviews[neg_index]);

                review_label[{1}] = 1;

                inputs.push_back(review_mat);
                labels.push_back(review_label);

                ++neg_index;
            }
        }
    }
}


void dataset::Review::get_training_set(
    std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    size_t all_samples = this->m_inputs.size();
    size_t train_end = static_cast<size_t>(all_samples * (1 - this->m_test_rate));

    lint index = 0;
    if (0 == limit)
    {
        limit = std::numeric_limits<lint>::max();
    }

    for (size_t i = 0; i < train_end; ++i)
    {
        inputs.push_back(this->m_inputs[i]);
        labels.push_back(this->m_labels[i]);
        ++index;
        if (index == limit)
        {
            break;
        }
    }
}


void dataset::Review::get_test_set(
    std::vector<neurons::TMatrix<>>& inputs, std::vector<neurons::TMatrix<>>& labels, lint limit) const
{
    size_t all_samples = this->m_inputs.size();
    size_t train_end = static_cast<size_t>(all_samples * (1 - this->m_test_rate));

    lint index = 0;
    if (0 == limit)
    {
        limit = std::numeric_limits<lint>::max();
    }

    for (size_t i = train_end; i < all_samples; ++i)
    {
        inputs.push_back(this->m_inputs[i]);
        labels.push_back(this->m_labels[i]);
        ++index;
        if (index == limit)
        {
            break;
        }
    }
}

