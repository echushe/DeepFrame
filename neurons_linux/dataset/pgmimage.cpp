/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 
 */

#include "pgmimage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// extern char *malloc();
// extern char *realloc();
// extern char *strcpy();

char * dataset::img_basename(const char * filename)
{
    char *new_;
    const char *part;
    int len, dex;

    len = strlen(filename);  dex = len - 1;
    while (dex > -1) {
        if (filename[dex] == '/') {
            break;
        }
        else {
            dex--;
        }
    }
    dex++;
    part = &(filename[dex]);
    len = strlen(part);
    // new_ = (char *)malloc((unsigned)((len + 1) * sizeof(char)));
    new_ = new char[len + 1];
    strcpy(new_, part);
    return(new_);
}


dataset::IMAGE * dataset::img_alloc()
{
    IMAGE *new_;

    // new_ = (IMAGE *)malloc(sizeof(IMAGE));
    new_ = new IMAGE;

    if (new_ == NULL) {
        printf("IMGALLOC: Couldn't allocate image structure\n");
        return (NULL);
    }
    new_->rows = 0;
    new_->cols = 0;
    new_->name = NULL;
    new_->data = NULL;
    return (new_);
}


dataset::IMAGE * dataset::img_creat(const char * name, int nr, int nc)
{
    int i, j;
    IMAGE *new_;

    new_ = img_alloc();
    // new_->data = (int *)malloc((unsigned)(nr * nc * sizeof(int)));
    new_->data = new int[nr * nc];

    new_->name = img_basename(name);
    new_->rows = nr;
    new_->cols = nc;
    for (i = 0; i < nr; i++) {
        for (j = 0; j < nc; j++) {
            img_setpixel(new_, i, j, 0);
        }
    }
    return (new_);
}


void dataset::img_free(IMAGE * img)
{
    // if (img->data) free((char *)img->data);
    // if (img->name) free((char *)img->name);
    // free((char *)img);

    if (img->data) delete[] img->data;
    if (img->name) delete[] img->name;
    
    delete img;
}


void dataset::img_setpixel(IMAGE * img, int r, int c, int val)
{
    int nc;

    nc = img->cols;
    img->data[(r * nc) + c] = val;
}


int dataset::img_getpixel(IMAGE * img, int r, int c)
{
    int nc;

    nc = img->cols;
    return (img->data[(r * nc) + c]);
}


dataset::IMAGE * dataset::img_open(const char * filename)
{
    IMAGE *new_;
    FILE *pgm;
    char line[512], intbuf[100], ch;
    int type, nc, nr, maxval, i, j, k, found;

    new_ = img_alloc();
    if ((pgm = fopen(filename, "r")) == NULL)
    {
        printf("IMGOPEN: Couldn't open '%s'\n", filename);
        return(NULL);
    }

    new_->name = img_basename(filename);

    /*** Scan pnm type information, expecting P5 ***/
    fgets(line, 511, pgm);
    sscanf(line, "P%d", &type);
    if (type != 5 && type != 2)
    {
        printf("IMGOPEN: Only handles pgm files (type P5 or P2)\n");
        fclose(pgm);
        return(NULL);
    }

    /*** Get dimensions of pgm ***/
    fgets(line, 511, pgm);
    sscanf(line, "%d %d", &nc, &nr);
    new_->rows = nr;
    new_->cols = nc;

    /*** Get maxval ***/
    fgets(line, 511, pgm);
    sscanf(line, "%d", &maxval);
    if (maxval > 255)
    {
        printf("IMGOPEN: Only handles pgm files of 8 bits or less\n");
        fclose(pgm);
        return(NULL);
    }

    // new_->data = (int *)malloc((unsigned)(nr * nc * sizeof(int)));
    new_->data = new int[nr * nc];
    
    if (new_->data == NULL)
    {
        printf("IMGOPEN: Couldn't allocate space for image data\n");
        fclose(pgm);
        return(NULL);
    }

    if (type == 5)
    {

        for (i = 0; i < nr; i++)
        {
            for (j = 0; j < nc; j++)
            {
                img_setpixel(new_, i, j, fgetc(pgm));
            }
        }

    }
    else if (type == 2) {

        for (i = 0; i < nr; i++)
        {
            for (j = 0; j < nc; j++)
            {

                k = 0;  found = 0;
                while (!found)
                {
                    ch = (char)fgetc(pgm);
                    if (ch >= '0' && ch <= '9')
                    {
                        intbuf[k] = ch;  k++;
                    }
                    else
                    {
                        if (k != 0)
                        {
                            intbuf[k] = '\0';
                            found = 1;
                        }
                    }
                }

                img_setpixel(new_, i, j, atoi(intbuf));

            }
        }

    }
    else
    {
        printf("IMGOPEN: Fatal impossible error\n");
        fclose(pgm);
        return (NULL);
    }

    fclose(pgm);
    return (new_);
}


int dataset::img_write(IMAGE * img, const char * filename)
{
    int i, j, nr, nc, k, val;
    FILE *iop;

    nr = img->rows;  nc = img->cols;
    iop = fopen(filename, "w");
    fprintf(iop, "P2\n");
    fprintf(iop, "%d %d\n", nc, nr);
    fprintf(iop, "255\n");

    k = 1;
    for (i = 0; i < nr; i++) {
        for (j = 0; j < nc; j++) {
            val = img_getpixel(img, i, j);
            if ((val < 0) || (val > 255)) {
                printf("IMG_WRITE: Found value %d at row %d col %d\n", val, i, j);
                printf("           Setting it to zero\n");
                val = 0;
            }
            if (k % 10) {
                fprintf(iop, "%d ", val);
            }
            else {
                fprintf(iop, "%d\n", val);
            }
            k++;
        }
    }
    fprintf(iop, "\n");
    fclose(iop);
    return (1);
}


