#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <regex.h>
#include <limits.h>

#ifndef MANDELBROT_LIB_H
#define MANDELBROT_LIB_H

struct BboxPiece{  //holds the piece-details (location in complex plane and the dimensions in pixels)
	mpfr_t* tqx;  //tq: third-quadrant, fq: first-quadrant
	mpfr_t* tqy;
	mpfr_t* fqx;
	mpfr_t* fqy;
	unsigned int wh;  //height and width in number of pixels (only square pieces to ease the management)
};

// Requires MPFR 4.1.0, Removed for now
//int decimal_digits(int precision_bits);

static void mpfr_array_clear_foo(mpfr_t* temp, int n);

static void mpfr_array_init_foo(mpfr_t* temp, unsigned int n, unsigned int* p);

static void compare_cutoff(long unsigned int* cutoffSquared, mpfr_t* temp, short* answer);

void single_iter(long unsigned int* cutoffSquared, long unsigned int* maxIter, mpfr_t* c_x, mpfr_t* c_y, mpfr_t* temp, unsigned int* result);

void piece_iter(long unsigned int* cutoffSquared, long unsigned int* maxIter, unsigned int* prec, struct BboxPiece* bb, unsigned int* resultArray);

int write_piece(char* filename, unsigned int* arr, char* header, unsigned int wh);

#endif