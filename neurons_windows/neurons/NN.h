#pragma once
#include "TMatrix.h"
#include "Functions.h"
#include "NN_layer.h"
#include "Dataset.h"
#include <iostream>
#include <random>

class NN
{
private:
    std::uniform_int_distribution<size_t> m_train_distribution;
    std::uniform_int_distribution<size_t> m_test_distribution;

protected:

    double m_l_rate;
    double m_mmt_rate;
    lint m_threads;

    std::string m_model_file;

    // The training set
    std::vector<neurons::TMatrix<>> m_train_set;
    // The training label
    std::vector<neurons::TMatrix<>> m_train_labels;
    // The test set
    std::vector<neurons::TMatrix<>> m_test_set;
    // The test label
    std::vector<neurons::TMatrix<>> m_test_labels;

    // The layers of neural network
    std::vector<std::shared_ptr<neurons::NN_layer>> m_layers;

public:
    NN(double l_rate, double mmt_rate, lint threads, const std::string & model_file, const dataset::Dataset &d_set);

    ~NN();

    NN(const NN & other) = delete;
    NN(NN && other) = delete;
    NN & operator = (const NN & other) = delete;
    NN & operator = (NN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const = 0;

    virtual void save_layers_as_images() const = 0;

    lint n_layers() const;

    void print_train_set(std::ostream & os) const;

    void print_train_label(std::ostream & os) const;

    void print_test_set(std::ostream & os) const;

    void print_test_label(std::ostream & os) const;

public:
    void train_network(
        lint batch_size,
        lint epoch_size,
        lint epochs,
        lint epochs_between_saves,
        lint secs_allowed);

    void test_network(lint batch_size, lint epoch_size);

    std::vector<neurons::TMatrix<>> network_predict(lint batch_size, const std::vector<neurons::TMatrix<>> & inputs) const;

    virtual bool load(const std::string & file_name) = 0;

    virtual bool load_until(const std::string & file_name, lint layer_index) = 0;

    virtual void initialize_model() = 0;

    virtual void save(const std::string & file_name) const = 0;

private:

    void get_batch(
        lint batch_size,
        std::vector<std::vector<neurons::TMatrix<>>> & data_batch,
        std::vector<std::vector<neurons::TMatrix<>>> & label_batch,
        const std::vector<neurons::TMatrix<>> & data,
        const std::vector<neurons::TMatrix<>> & label,
        std::uniform_int_distribution<size_t> & distribution);


    double train_step(
        lint batch_size,
        const std::vector<std::vector<neurons::TMatrix<>>> & inputs,
        const std::vector<std::vector<neurons::TMatrix<>>> & targets,
        std::vector<std::vector<neurons::TMatrix<>>> & preds);

    double test_step(
        lint batch_size,
        const std::vector<std::vector<neurons::TMatrix<>>> & inputs,
        const std::vector<std::vector<neurons::TMatrix<>>> & targets,
        std::vector<std::vector<neurons::TMatrix<>>> & preds);

    std::vector<std::vector<neurons::TMatrix<>>> predict_step(
        lint batch_size,
        const std::vector<std::vector<neurons::TMatrix<>>> & inputs) const;

    double get_accuracy(
        const neurons::TMatrix<> & pred, const neurons::TMatrix<> & target);

    double get_accuracy(
        lint batch_size,
        const std::vector<std::vector<neurons::TMatrix<>>> & preds,
        const std::vector<std::vector<neurons::TMatrix<>>> & targets);

    virtual std::vector<neurons::TMatrix<>> test(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id) = 0;

    virtual std::vector<neurons::TMatrix<>> optimise(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id) = 0;

    virtual std::vector<neurons::TMatrix<>> predict(
        const std::vector<neurons::TMatrix<>> & inputs,
        lint thread_id) const = 0;

};

std::ostream & operator << (std::ostream & os, const NN & nn);


