/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 * 28-05-2018 Chunnan Sheng, University of New South Wales
 *      Convert to C++ mode
 *
 ******************************************************************
 */

#pragma once

namespace dataset
{
    struct IMAGE
    {
        char *name;
        int rows, cols;
        int *data;
    };


    /*** User accessible macros ***/

#define ROWS(img)  ((img)->rows)
#define COLS(img)  ((img)->cols)
#define NAME(img)   ((img)->name)

/*** User accessible functions ***/

    IMAGE *img_open(const char * filename);

    IMAGE *img_creat(const char * name, int nr, int nc);

    void img_setpixel(IMAGE * img, int r, int c, int val);

    int img_getpixel(IMAGE * img, int r, int c);

    int img_write(IMAGE * img, const char * filename);

    void img_free(IMAGE * img);

    char * img_basename(const char * filename);

    IMAGE * img_alloc();

    IMAGE * img_creat(const char * name, int nr, int nc);


}
