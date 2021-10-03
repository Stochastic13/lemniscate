#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <regex.h>
#include <limits.h>
#include "mandelbrot_lib.h"

/*
mode: sp - single piece
	argv:
		1. "sp"
		2. prec
		3. wh
		4. maxiter
		5. cutoff
		6. filename
		7. header information (without spaces)
		8. tqx
		9. tqy
		10. fqx
		11. fqy
mode: mr - parallel multiple runs (OMPI; run with mpirun)
	argv:
		1. "mr"
		2. file (configuration options)
mode: cl - collate
	argv:
		1. "cl"
		2. n (number of pieces to look for in the given path)
		3. path
		4. output file name
mode: spf - single piece with config file
	argv same as mr mode
mode: dpn - digit-precision number
	argv: all arguments are the integer precision values to query for
*/
int  main(int argc, char** argv){
	if (strcmp(*(argv + 1), "sp") == 0){
		struct BboxPiece bb;
		int prec = atoi(*(argv + 2));
		unsigned int prec2 = (unsigned int)prec;
		mpfr_t tqx, tqy, fqx, fqy;
		mpfr_init2(tqx, prec);
		mpfr_init2(tqy, prec);
		mpfr_init2(fqx, prec);
		mpfr_init2(fqy, prec);
		mpfr_set_str(tqx, *(argv + 8), 10, MPFR_RNDN);
		mpfr_set_str(tqy, *(argv + 9), 10, MPFR_RNDN);
		mpfr_set_str(fqx, *(argv + 10), 10, MPFR_RNDN);
		mpfr_set_str(fqy, *(argv + 11), 10, MPFR_RNDN);
		bb.tqx = &tqx;
		bb.tqy = &tqy;
		bb.fqx = &fqx;
		bb.fqy = &fqy;
		bb.wh = (unsigned int)strtol(*(argv + 3), NULL, 10);
		long unsigned int maxIter = strtoul(*(argv + 4), NULL, 10);
		long unsigned int cutoffSquared = strtoul(*(argv + 5), NULL, 10);
		cutoffSquared = cutoffSquared * cutoffSquared;
		unsigned int* resultArray = (unsigned int*)malloc((bb.wh) * (bb.wh) * sizeof(unsigned int));
		piece_iter(&cutoffSquared, &maxIter, &prec2, &bb, resultArray);
		write_piece(*(argv + 6), resultArray, *(argv + 7), bb.wh);
		free(resultArray);
	}
	if (strcmp(*(argv + 1), "dpn") == 0){
		for (int i = 0; i < argc - 2; i++){
//          Needs MPFR 4.1.0. Removed for now
//			printf("A precision of %s gives you %d decimal digits\n", *(argv + 2 + i), decimal_digits(atoi(*(argv + 2 + i))));
			printf("A precision of %s gives you %d decimal digits\n", *(argv + 2 + i), 0);

		}
	}
	if (strcmp(*(argv + 1), "mr") == 0){
		FILE* f;
		f = fopen(*(argv + 2), "r");
		fclose(f);
	}
}
