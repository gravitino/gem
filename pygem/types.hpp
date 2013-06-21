#ifndef __TYPES_H__
#define __TYPES_H__

#include <vector>                     // stl vector

/*******************************************************************************
* type definitions
*******************************************************************************/
typedef unsigned int itype;
typedef float ftype;

typedef std::vector<ftype> series;
typedef std::vector<itype> coords;
typedef std::pair<itype, ftype> pair; // for sorting the result

/*******************************************************************************
* match definitions
*******************************************************************************/
typedef struct {

    itype left, right, length;        // left, right border and length of match
    ftype penalty;                    // mean penalty of match

} single_match;

typedef std::vector<single_match> result;
#endif
