#include "Fibonacci.h"



misc::Fibonacci::Fibonacci()
{
}


misc::Fibonacci::~Fibonacci()
{
}

lint misc::Fibonacci::operator()(int index)
{
    neurons::TMatrix<lint> fib_mat{ neurons::Shape{2, 2} };
    fib_mat[{0, 0}] = 1;
    fib_mat[{0, 1}] = 1;
    fib_mat[{1, 0}] = 1;
    fib_mat[{1, 1}] = 0;

    neurons::TMatrix<lint> ret = neurons::matrix_pow(fib_mat, index + 1);

    return ret[{1, 1}];
}
