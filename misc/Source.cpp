#include "Fibonacci.h"
#include <iostream>

int main()
{
    misc::Fibonacci fib;

    for (int i = 0; i < 80; ++i)
    {
        std::cout << fib(i) << "\n";
    }

    return 0;
}