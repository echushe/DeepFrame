#pragma once
#include <string>

namespace neurons
{
    /*
    Definitions of most frequently used exceptions
    */

    const std::string invalid_shape_dim("Shape: Shape dimensions should not be less than 1");
    const std::string invalid_shape_val("Shape: Each dimension of Shape should not be less than 1");
    const std::string invalid_coord_dim("Coordinate: coodinate dimensions should not be less than 1");
    const std::string invalid_coord_val("Coordinate: Each dimension of coordinate should not be less than 0");
    const std::string invalid_coordinate("Coordinate: invalid coordinate dimentsions or values");
    const std::string invalid_shape("Matrix: The two matrices should have the same shape");
    const std::string incompatible_shape("Matrix: The two matrices cannot be multiplied due to inconsistent shapes");
    const std::string incompatible_size("Matrix: The two matrices should have the same amount of elements");
}
