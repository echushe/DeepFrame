#pragma once

#include "TMatrix.h"
#include <vector>


namespace dataset
{
    typedef long long int lint;

    /*
    This is an abstract interface of Dataset to read inputs and labels
    */
    class Dataset
    {
    public:
        virtual void get_training_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const = 0;

        virtual void get_test_set(
            std::vector<neurons::TMatrix<>> & inputs, std::vector<neurons::TMatrix<>> & labels, lint limit = 0) const = 0;
    };
}
