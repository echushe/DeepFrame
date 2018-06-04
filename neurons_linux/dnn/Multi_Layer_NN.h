/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/

#pragma once
#include "NN.h"
#include "FCNN_layer.h"
#include <iostream>


class Multi_Layer_NN : public NN
{
private:

    lint m_input_size;
    lint m_output_size;

public:
    // Create Multi_layer_NN
    Multi_Layer_NN(
        double l_rate,
        double mmt_rate,
        lint threads,
        const std::string & model_file,
        const dataset::Dataset &d_set);

    // Copies and moves are prohibited

    Multi_Layer_NN(const Multi_Layer_NN & other) = delete;
    Multi_Layer_NN(Multi_Layer_NN && other) = delete;
    Multi_Layer_NN & operator = (const Multi_Layer_NN & other) = delete;
    Multi_Layer_NN & operator = (Multi_Layer_NN && other) = delete;

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

