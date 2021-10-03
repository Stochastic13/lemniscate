#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <regex.h>
#include <limits.h>

#ifndef COREFUNC_H
#define COREFUNC_H

struct BboxPiece{  //holds the piece-details (location in complex plane and the dimensions in pixels)
	mpfr_t* tqx;  //tq: third-quadrant, fq: first-quadrant
	mpfr_t* tqy;
	mpfr_t* fqx;
	mpfr_t* fqy;
	unsigned int wh;  //height and width in number of pixels (only square pieces to ease the management)
};

static void mpfr_array_clear_foo(mpfr_t* temp, int n);

static void mpfr_array_init_foo(mpfr_t* temp, unsigned int n, unsigned int* p);

static void compare_cutoff(long unsigned int* cutoffSquared, mpfr_t* temp, short* answer);

static void single_iter(long unsigned int* cutoffSquared, long unsigned int* maxIter, mpfr_t* c_x, mpfr_t* c_y, mpfr_t* temp, unsigned int* result);

void piece_iter(long unsigned int* cutoffSquared, long unsigned int* maxIter, unsigned int* prec, struct BboxPiece* bb, unsigned int* resultArray);

void write_piece(char* filename, unsigned int* arr, char* header, unsigned int wh);

static void parser(char* filename, unsigned int* prec, unsigned int* maxIter, long unsigned int* cutoff, unsigned int* whglobal, char** bb, char* misc);

static void regex_checker(char* filename);

void worker_foo();

void manager_foo(char* filename, int world_size);

#endif