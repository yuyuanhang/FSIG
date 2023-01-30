/**
 * @file defines.h
 * @brief contains defines (1) type define; (2) macro define
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */
#ifndef DEFINES_H
#define DEFINES_H

#include <chrono> // NOLINT [build/c++11]
#include <vector>
#include <utility>

// type define
typedef unsigned Vertex;
typedef unsigned Edge;
typedef int Label;
typedef std::chrono::high_resolution_clock::time_point TimePoint;

// macro define
#define INVALID_LABEL 0xffffffff
#define DEBUG 1
//#define STORE 1
#define READ 1
#define FULL_MASK 0xffffffff
#define INVALID_DEGREE 0xffffffff
#define INVALID 0xffffffff
#define HASHSEED 17
#define SIGLEN 64*8
#define SIGNUM 16

#endif