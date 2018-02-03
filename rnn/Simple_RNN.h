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
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    // Copies and moves are prohibited

    Simple_RNN(const Simple_RNN & other) = delete;
    Simple_RNN(Simple_RNN && other) = delete;
    Simple_RNN & operator = (const Simple_RNN & other) = delete;
    Simple_RNN & operator = (Simple_RNN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

private:

    virtual std::vector<neurons::Matrix> test(
        const std::vector<neurons::Matrix> & inputs,
        const std::vector<neurons::Matrix> & targets,
        lint thread_id);

    virtual std::vector<neurons::Matrix> optimise(
        const std::vector<neurons::Matrix> & inputs,
        const std::vector<neurons::Matrix> & targets,
        lint thread_id);
};



