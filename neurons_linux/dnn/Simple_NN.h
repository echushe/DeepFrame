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

public:
    Simple_NN(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    Simple_NN(const Simple_NN & other) = delete;
    Simple_NN(Simple_NN && other) = delete;
    Simple_NN & operator = (const Simple_NN & other) = delete;
    Simple_NN & operator = (Simple_NN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

private:

    virtual std::vector<neurons::TMatrix<>> test(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

    virtual std::vector<neurons::TMatrix<>> optimise(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

};


