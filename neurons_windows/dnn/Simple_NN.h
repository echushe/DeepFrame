/********************************************************************

Programmed by Chunnan Sheng

A simple NN without any hidden layers

*********************************************************************/

#pragma once
#include "NN.h"
#include "FCNN_layer.h"
#include <iostream>

class Simple_NN : public NN
{
private:

    lint m_input_size;
    lint m_output_size;

public:
    Simple_NN(
        double l_rate,
        double mmt_rate,
        lint threads,
        const std::string & model_file,
        const dataset::Dataset &d_set);

    Simple_NN(const Simple_NN & other) = delete;
    Simple_NN(Simple_NN && other) = delete;
    Simple_NN & operator = (const Simple_NN & other) = delete;
    Simple_NN & operator = (Simple_NN && other) = delete;

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


