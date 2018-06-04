/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/

#pragma once
#include "NN.h"
#include <iostream>


class Simple_RNN : public NN
{
private:

public:
    Simple_RNN(
        double l_rate,
        double mmt_rate,
        lint threads,
        const std::string & model_file,
        const dataset::Dataset &d_set);

    // Copies and moves are prohibited

    Simple_RNN(const Simple_RNN & other) = delete;
    Simple_RNN(Simple_RNN && other) = delete;
    Simple_RNN & operator = (const Simple_RNN & other) = delete;
    Simple_RNN & operator = (Simple_RNN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

    virtual void save_layers_as_images() const;

    virtual bool load(const std::string & file_name);

    virtual bool load_until(const std::string & file_name, lint layer_index);

    virtual void initialize_model();

    virtual void save(const std::string & file_name) const;

private:

    virtual std::vector<neurons::TMatrix<>> test(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

    virtual std::vector<neurons::TMatrix<>> optimise(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

    virtual std::vector<neurons::TMatrix<>> predict(
        const std::vector<neurons::TMatrix<>> & inputs,
        lint thread_id) const;
};



