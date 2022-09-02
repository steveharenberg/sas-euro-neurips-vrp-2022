
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

#if defined (__GNUC__)
#  define INLINE inline __attribute__((always_inline))
#elif defined (_MSC_VER)
#  define INLINE __forceinline
#else
#  define INLINE inline
#endif


class Matrix
{
    int cols_;                  // The number of columns of the matrix
    int** data_;     // The vector where all the data is stored (this represents the matrix)
    int* data__;
public:
    // Empty constructor: with zero columns and a vector of size zero
    Matrix() : cols_(0), data_(nullptr)
    {}

    // Constructor: create a matrix of size dimension by dimension, using a C++ vector of size dimension * dimension 
    Matrix(const int dimension) : cols_(dimension)
    {
        data__= new int[dimension*dimension];
        data_ = new int*[dimension];
        for(int i=0; i<dimension; ++i){
          data_[i] = data__+dimension*i;
        }
    }

    // Set a value val at position (row, col) in the matrix
    INLINE void set(const int row, const int col, const int val)
    {
        // data_[cols_ * row + col] = val;
        data_[row][col] = val;
    }

    // Get the value at position (row, col) in the matrix
    INLINE int get(const int row, const int col) const
    {
        // return data_[cols_ * row + col];
        return data_[row][col];
    }
};

// Implementation of a matrix in a C++ vector
// This class is used because a flat vector is faster than a vector of vectors which requires two lookup operations rather than one to index a matrix element
class Matrix_old
{
    int cols_;                  // The number of columns of the matrix
    std::vector<int> data_;     // The vector where all the data is stored (this represents the matrix)

public:
    // Empty constructor: with zero columns and a vector of size zero
    Matrix_old() : cols_(0), data_(std::vector<int>(0))
    {}

    // Constructor: create a matrix of size dimension by dimension, using a C++ vector of size dimension * dimension 
    Matrix_old(const int dimension) : cols_(dimension)
    {
        data_ = std::vector<int>(dimension * dimension);
    }

    // Set a value val at position (row, col) in the matrix
    INLINE void set(const int row, const int col, const int val)
    {
        data_[cols_ * row + col] = val;
    }

    // Get the value at position (row, col) in the matrix
    INLINE int get(const int row, const int col) const
    {
        return data_[cols_ * row + col];
    }
};

#endif
