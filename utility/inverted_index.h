/**
 * @file inverted.h
 * @brief contains two logic classes InvertedIndex and InversedFunction
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include <vector>

using std::vector;

template<class T,class N>
class InvertedIndex {
public:
    N length_;
    T num_keys_;
    N* index_;
    N* list_;

    InvertedIndex(N length, T num_keys, T* forward_index) { 
        num_keys_ = num_keys;
        length_ = length;

        vector<vector<T>> key_bucket(num_keys);
        for (N i = 0; i < length; i++) {
            key_bucket[forward_index[i]].push_back(i);
        }

        list_ = new N[length];
        index_ = new N[num_keys+1];
        index_[0] = 0;
        for (N i = 0; i < num_keys; i++) {
            index_[i+1] = index_[i] + key_bucket[i].size();
            copy(key_bucket[i].begin(), key_bucket[i].end(), list_+index_[i]);
        }
    }

    ~InvertedIndex() { delete[] index_; delete[] list_; }
    N num_values(T key) { return index_[key+1] - index_[key]; }
};

/*!
* A logic class that length is fixed at the beginning
*/
template<class T>
class InversedFunction {
public:
    T length_;
    vector<T> inversed_f_;

    InversedFunction (T length) {
        length_ = length;
        inversed_f_.resize(length);
    }

    ~InversedFunction() { delete[] inversed_f_; }
    void add_x(T x, T fx) { inversed_f_[fx] = x; }
};

#endif