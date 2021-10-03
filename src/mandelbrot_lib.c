#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <regex.h>
#include <limits.h>
#include "mandelbrot_lib.h"

/*
Throughout the code, Re(z) and z_x are used interchangeably. The same for Im(z) and z_y.
z_new refers to the z_(i + 1) when used in context with z (referring to z_(i)) for an iteratively updated variable.
|z| is the absolute value of the complex number z.
Currently hard-coded to use 32-bit (unsigned int) values for the final array (upto 4294967295).
For larger max-iterations or 32-bit machines, beware of overflow (currently quits with an error).
As mentioned in the ReadMe, overuse of pointers is intentional. :D
*/

#define RN MPFR_RNDN // for easy typing later on

// Given a bit-precision value, return the number of significant decimal digits supported at that precision level
// Requires MPFR 4.1.0, Removed for now
//int decimal_digits(int precision_bits){
//	return mpfr_get_str_ndigits(10, precision_bits);
//}

// free the memory from a mpfr_t array with n elements
static void mpfr_array_clear_foo(mpfr_t* temp, int n){
	for (int i = 0; i < n; i++){
		mpfr_clear(*(temp + i));
	}
}

// initialize a mpfr_t array with n elements with a given precision
static void mpfr_array_init_foo(mpfr_t* temp, unsigned int n, unsigned int* p){
	for (int i = 0; i < n; i++){
		mpfr_init2(*(temp + i), *p);
	}
}

/* given a complex number z = z_x + z_y * i, this function compares |z| 
with the integer cutoff and returns 1 is |z| > c and 0 otherwise */
static void compare_cutoff(long unsigned int* cutoffSquared, mpfr_t* temp, short* answer){
	//assumes that temp[0] and temp[1] hold Re(z) and Im(z) respectively
	mpfr_mul(*(temp + 2), *temp, *temp, RN);  //store z_x^2 in temp[2]
	mpfr_mul(*(temp + 3), *(temp + 1), *(temp + 1), RN);  //store z_y^2 in temp[3]
	mpfr_add(*(temp + 4), *(temp + 2), *(temp + 3), RN);  //store z_x^2 + z_y^2 = |z|^2 in temp[4]
	if (mpfr_cmp_ui(*(temp + 4), *cutoffSquared) > 0){  //if |z|^2 > cutoff^2 (avoiding sqrt)
		*answer = 1;  //blowup
	}
	else {
		*answer = 0;  //probable candidate for the mandelbrot set
	}
}

/* given a complex number c = c_x + c_y * i, this function iterates the main mandelbrot equation (z = z^2 + c)
starting from z = 0 + 0i and stores the iteration count when z surpasses cutoff (or iteration count hits maxIter)*/
void single_iter(long unsigned int* cutoffSquared, long unsigned int* maxIter, mpfr_t* c_x, mpfr_t* c_y, mpfr_t* temp, unsigned int* result){
	long unsigned int i = 0;  //current iteration count (can never be zero in the final result)
	short check = 0;  //to hold whether the given c causes a blowup before maxIter
	mpfr_set_ui(*temp, 0, RN);  //initialize z = 0 + 0i (temp[0] holds Re(z) and temp[1] holds Im(z))
	mpfr_set_ui(*(temp + 1), 0, RN);
	mpfr_mul(*(temp + 2), *temp, *temp, RN);  //store z_x^2 in temp[2]
	mpfr_mul(*(temp + 3), *(temp + 1), *(temp + 1), RN); //store z_y^2 in temp[3]
	while (i < *maxIter){
		i++;
		mpfr_sub(*(temp + 2), *(temp + 2), *(temp + 3), RN); //store z_x^2 - z_y^2 in temp[2]
		mpfr_add(*(temp + 2), *(temp + 2), *c_x, RN);  //store z_x^2 - z_y^2 + c_x = z_new_x in temp[2]
		mpfr_mul(*(temp + 3), *temp, *(temp + 1), RN);  //store z_x * z_y in temp[3]
		mpfr_mul_ui(*(temp + 3), *(temp + 3), 2, RN);  //store 2 * z_x * z_y in temp[3]
		mpfr_add(*(temp + 3), *(temp + 3), *c_y, RN);  //2 * z_x * z_y + c_y = z_new_y in temp[3]
		mpfr_set(*temp, *(temp + 2), RN);  //set temp and temp[1] to z_new_x and z_new_y respectively
		mpfr_set(*(temp + 1), *(temp + 3), RN);
		compare_cutoff(cutoffSquared, temp, &check);  //compare |z_new| to cutoff and load z_new_x^2 and z_new_y^2 in temp[2] and temp[3]
		if (check) {
			break;  //end the iteration if the cutoff value reached
		}
	}
	*result = i;
}

/* given a bbox (region in complex plane) this function iterates a lattice of uniformly spaced points (pixels)
in the region and stores the value of the iteration count for blowup (or the maximum iteration value) */
void piece_iter(long unsigned int* cutoffSquared, long unsigned int* maxIter, unsigned int* prec, struct BboxPiece* bb, unsigned int* resultArray){
	mpfr_t* temp = (mpfr_t*)malloc(5 * sizeof(mpfr_t));
	mpfr_array_init_foo(temp, 5, prec);
	mpfr_t diff;  //store the size of the jump per pixel (i.e. the spacing between the pixels)
	mpfr_init2(diff, *prec);
	mpfr_sub(diff, *(bb->fqx), *(bb->tqx), RN);  //assuming the jump sizes are equal in both x and y directions
	mpfr_div_ui(diff, diff, bb->wh, RN);
	mpfr_t* c = (mpfr_t*)malloc(2 * sizeof(mpfr_t));  //hold the current c value as c_x = c[0] and c_y as c[1]
	mpfr_array_init_foo(c, 2, prec);
	mpfr_set(*(c + 1), *(bb->tqy), RN);  //set c_y to the third-quadrant-y value
	long int counter = 0;
	for (unsigned int i = 0; i < (bb->wh); i++){ //iterate along y-values
		mpfr_set(*c, *(bb->tqx), RN);  //set c_x to the third-quadrant-x value
		for (unsigned int j = 0; j < (bb->wh); j++){  //iterate along x-values
			single_iter(cutoffSquared, maxIter, c, c + 1, temp, resultArray + counter);  //compute the result for the single pixel
			mpfr_add(*c, *c, diff, RN);  //increment the x-value
			counter++;
		}
		mpfr_add(*(c + 1), *(c + 1), diff, RN);  //increment the y-value
	}
}

/* given a result-array, this function outputs it to a tab-separated file called filename */
int write_piece(char* filename, unsigned int* arr, char* header, unsigned int wh){
	FILE* f;
	f = fopen(filename, "w");
	if (f == NULL){
		printf("%s\n", "C ERROR: File couldn't be opened.");
		return 1;
	}
	fprintf(f, "%s\n", header);  //header has the identifying information of the piece and the region corresponding to it
	long int counter = 0;
	for (unsigned int i = 0; i < wh; i++){  // sequentially output the array
		for (unsigned int j = 0; j < wh; j++){
			fprintf(f, "%u\t", *(arr + counter));
			if (ferror(f)){
				printf("%s\n", "C ERROR: Writing to file failed.");
				return 1;
			}
			counter++;
		}
		fputs("\n", f);
		if (ferror(f)){
			printf("%s\n", "C ERROR: Writing to file failed.");
			return 1;
		}
	}
	fclose(f);
	return 0;
}

/* Main handler to call the required functions and iterate a piece as required. Also manages the return value
in accordance with what the python wrapper requires. This is where the memory is allocated. Can be used standalone
with command line arguments too (hence the "argc" and "argv")
	argv:
		1. prec (cast to int)
		2. wh (cast to unsigned int)
		3. maxiter (cast to long unsigned int)
		4. cutoff (cast to long unsigned int)
		5. filename (string: arbitrarily long)
		6. header information without spaces (string: arbitrarily long)
		7. tqx (string: arbitrarily long)
		8. tqy (string: arbitrarily long)
		9. fqx (string: arbitrarily long)
		10. fqy (string: arbitrarily long)
*/
int handler(int argc, char** argv){
	struct BboxPiece bb;
	long int prec = strtol(*(argv + 2), NULL, 10); // to feed to mpfr_init2
	if ((prec < MPFR_PREC_MIN) || (prec > (MPFR_PREC_MAX - 32))){  // 32 is for extra safety (see MPFR docs)
		printf("%s%d%d\n", "C ERROR: Precision entered is incorrect. Min and Max values supported are: ", MPFR_PREC_MIN, MPFR_PREC_MAX);
		return -1;
	}
	unsigned int prec2 = (unsigned int)prec;  // for use later in the code
	mpfr_t tqx, tqy, fqx, fqy;
	mpfr_init2(tqx, prec);  // initialize the main 4 bounding box corners
	mpfr_init2(tqy, prec);
	mpfr_init2(fqx, prec);
	mpfr_init2(fqy, prec);
	mpfr_set_str(tqx, *(argv + 7), 10, MPFR_RNDN);  // set the bounding box corners to appropriate values
	mpfr_set_str(tqy, *(argv + 8), 10, MPFR_RNDN);
	mpfr_set_str(fqx, *(argv + 9), 10, MPFR_RNDN);
	mpfr_set_str(fqy, *(argv + 10), 10, MPFR_RNDN);
	bb.tqx = &tqx;
	bb.tqy = &tqy;
	bb.fqx = &fqx;
	bb.fqy = &fqy;
	bb.wh = (unsigned int)strtol(*(argv + 2), NULL, 10);
	long unsigned int maxIter = strtoul(*(argv + 3), NULL, 10);
	long unsigned int cutoffSquared = strtoul(*(argv + 4), NULL, 10);
	cutoffSquared = cutoffSquared * cutoffSquared;  // to compare with |z|^2
	unsigned int* resultArray = (unsigned int*)malloc((bb.wh) * (bb.wh) * sizeof(unsigned int));
	piece_iter(&cutoffSquared, &maxIter, &prec2, &bb, resultArray);
	int result_write = write_piece(*(argv + 5), resultArray, *(argv + 6), bb.wh);
	free(resultArray);
	if (result_write){
		return -2;
	}
	return 0;
}
