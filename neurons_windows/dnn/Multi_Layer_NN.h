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

public:
    Multi_Layer_NN(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    // Copies and moves are prohibited

    Multi_Layer_NN(const Multi_Layer_NN & other) = delete;
    Multi_Layer_NN(Multi_Layer_NN && other) = delete;
    Multi_Layer_NN & operator = (const Multi_Layer_NN & other) = delete;
    Multi_Layer_NN & operator = (Multi_Layer_NN && other) = delete;

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

