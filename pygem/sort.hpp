#ifndef __SORT_H__
#define __SORT_H__

#include <map>                        // stl map
#include <algorithm>                  // stl sort

/*******************************************************************************
* sort related stuff
*******************************************************************************/

// key function for sorting the result
bool sort_pred(const pair& left, const pair& right) {
        return left.second < right.second;
}

// key function for non-overlapping match
struct cmp_interval {
    bool operator()(const single_match& a, const single_match& b) {

        const ftype p = 0.5;
        const ftype diffc = ((ftype)(a.left+a.right))-((ftype)(b.left+b.right));
        const ftype diffa = ((ftype)a.right)-((ftype)a.left);
        const ftype diffb = ((ftype)b.right)-((ftype)b.left);
        const ftype min2  = 2*std::min(diffa, diffb);

        if ((diffa+diffb-ABS(diffc))/min2 > p)
            return false;

        return a.left+a.right < b.left+b.right;
    }
};

// remove overlapping results ordered by penalty with tree
int non_overlap(result *R) {

    // early exit for empty result
    if (R->size() == 0)
        return 0;

    // temporary storage for results
    std::vector<single_match> *temp = new std::vector<single_match>();

    // interval tree for efficient collision detection
    std::map<single_match, bool, cmp_interval> Map =
    std::map<single_match, bool, cmp_interval>();

    // now sweep over the result
    for (itype i = 0; i < R->size(); ++i) {
        if(Map[R->at(i)] == false && R->at(i).left != NIL) {
            Map[R->at(i)] = true;
            temp->push_back(R->at(i));
        }
    }

    // clear result for sweeped result
    R->clear();

    // copy sweeped result to result
    for (itype i = 0; i < temp->size(); ++i)
        R->push_back(temp->at(i));

    // free ram
    delete temp;

    // successful
    return 0;
}

#endif
