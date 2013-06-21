#include "gem.hpp"
#include "sort.hpp"

/*******************************************************************************
* matching core
*******************************************************************************/

template <bool omp, bool squared> 
int meta_match (series *N, series *H, result *R, itype St0, itype St1, ftype E) {

    const itype Nsize = N->size();
    const itype Hsize = H->size();
    const itype Msize = (St0+1)*Hsize;

    itype *sou = new itype[Msize];
    itype *len = new itype[Msize];
    ftype *pen = new ftype[Msize];
    ftype *a_0 = new ftype[Msize];
    
    ftype *E_k = new ftype[St0];
    ftype *E_l = new ftype[St1];
    
    for (itype k = 0; k < St0; ++k)
        E_k[k] = 1-pow(1-E, 1.0/(k+1));
    
    for (itype l = 0; l < St1; ++l)
        E_l[l] = 1-pow(1-E, l+1);
    
    if (!omp)
        omp_set_num_threads(1);
        
    # pragma omp parallel
    {

        if (omp_get_thread_num() == 0)
            std::cout << "Running on " << omp_get_num_threads() << " CPU threads\n";

        # pragma omp for nowait
        for (itype j = 0; j < Hsize; ++j) {
            sou[j] = j;
            len[j] = 1;
            pen[j] = 0;
            a_0[j] = H->at(j)-N->at(0);    
        }
    
        for (itype i = 1; i < St0+1; ++i) {
            # pragma omp for nowait
            for (itype j = 0; j < Hsize; ++j) {

                const itype dest = i*Hsize+j;
                
                sou[dest] = NIL;
                len[dest] = 0;
                pen[dest] = INFTY;  
            }
        }
    
        # pragma omp barrier
        for (itype i = 0; i < Nsize-1; ++i) {

            const itype del_ptr = Hsize*((i+St0)%(St0+1));
            const itype src_ptr = Hsize*(i%(St0+1));
            const itype trg_ptr = ((i+1)%(St0+1))*Hsize;

            # pragma omp for
            for (itype j = 0; j < Hsize; ++j) {

                const itype del = del_ptr+j;                

                sou[del] = NIL;
                len[del] = 0;
                pen[del] = INFTY;
            }

            const itype border_k = std::min(St0, Nsize-i-1);

            # pragma omp for 
            for (itype j = 0; j < Hsize-1; ++j) {
                
                const itype src = src_ptr+j;   
              
                for (itype k = 0; k < border_k; ++k) {

                    const itype trg_ptr = ((i+1+k)%(St0+1))*Hsize;
    
                    const itype i1k = i+1+k, j1 = j+1;
                    const itype trg = trg_ptr+j1;
                
                    ftype a0 = H->at(j1)-N->at(i1k);
                    a0 = (1-E_k[k])*a_0[src]+E_k[k]*a0;
                
                    ftype dst = H->at(j1)-a0-N->at(i1k);
                   
                    if (squared)
                        dst *= dst;
                    else
                        dst = ABS(dst);
                        
                    if (pen[src]+dst < pen[trg]/len[trg]*(len[src]+1)) {
                    
                         sou[trg] = sou[src];
                         len[trg] = len[src]+1;
                         pen[trg] = pen[src]+dst;
                         a_0[trg] = a0;
                    }    
                }
            
                const itype border_l = std::min(St1, Hsize-j-1);

                for (itype l = 1; l < border_l; ++l) {
                
                    const itype i1 = i+1, j1l = j+1+l;
                    const itype trg = trg_ptr+j1l;

                    ftype a0 = H->at(j1l)-N->at(i1);
                    a0 = (1-E_l[l])*a_0[src]+E_l[l]*a0;
    
                    ftype dst = H->at(j1l)-a0-N->at(i1);
                    
                    if (squared)
                        dst *= dst;
                    else
                        dst = ABS(dst);
                   
                    if (pen[src]+dst < pen[trg]/len[trg]*(len[src]+1)) {
                    
                        sou[trg] = sou[src];
                        len[trg] = len[src]+1;
                        pen[trg] = pen[src]+dst;
                        a_0[trg] = a0;
                    }            
                }
            }
        }    
    } // end pragma omp parallel

    const itype lst_ptr = ((Nsize-1)%(St0+1))*Hsize;
    
    std::vector<pair> *hitlist = new std::vector<pair>();
    
    for (itype j = 0; j < H->size(); ++j)
        hitlist->push_back(pair(j, pen[lst_ptr+j]/len[lst_ptr+j]));

    std::sort(hitlist->begin(), hitlist->end(), sort_pred);

    for (itype j = 0; j < hitlist->size(); ++j) {

        single_match entry;
        entry.left    = sou[lst_ptr+hitlist->at(j).first];
        entry.right   = hitlist->at(j).first;
        entry.length  = len[lst_ptr+hitlist->at(j).first];
        entry.penalty = pen[lst_ptr+hitlist->at(j).first] / entry.length;

        R->push_back(entry);
    }

    non_overlap(R);

    delete hitlist;
   
    delete[] sou;
    delete[] len;
    delete[] pen;
    delete[] a_0;
    
    delete[] E_k;
    delete[] E_l;

    return 0;
}

int match(series* N, series* H, result *R, 
          itype St0, itype St1, ftype E, bool omp, bool squared) {

    if (omp) {
        if (squared)
            return meta_match <true, true> (N, H, R, St0, St1, E);
        else
            return meta_match <true, false> (N, H, R, St0, St1, E);
    } else {
        if (squared)
            return meta_match <false, true> (N, H, R, St0, St1, E);
        else
            return meta_match <false, false> (N, H, R, St0, St1, E);
    }
}

/*******************************************************************************
* backtracing core
*******************************************************************************/

template<bool omp, bool squared>
int meta_backtrace(series *N, series *H, single_match *M, series *L, 
                   coords *X, series *Y, itype St0, itype St1, ftype E) {

    if (N->size() < 2 || H->size() < 2)
        return 1;

    if (M->left == NIL)
          return 1;

    const itype width = M->right-M->left+1;
    const itype Nsize = N->size(), Msize = N->size()*width;
    
    itype *len = new itype[Msize];
    itype *p_i = new itype[Msize];
    itype *p_j = new itype[Msize];
    ftype *pen = new ftype[Msize];
    ftype *a_0 = new ftype[Msize];
    
    ftype *E_k = new ftype[St0];
    ftype *E_l = new ftype[St1];
    
   
    for (itype k = 0; k < St0; ++k)
        E_k[k] = 1-pow(1-E, 1.0/(k+1));
    
    for (itype l = 0; l < St1; ++l)
        E_l[l] = 1-pow(1-E, l+1);

    if (!omp)
        omp_set_num_threads(1);

    # pragma omp parallel
    {
   
        # pragma omp for nowait
        for (itype j = 0; j < width; ++j) {
            len[j] = 1;
            p_i[j] = 0;
            p_j[j] = 0;
            pen[j] = 0;
            a_0[j] = H->at(j+M->left)-N->at(0);
        }
    
        for (itype i = 1; i < Nsize; ++i) {
            # pragma omp for nowait
            for (itype j = 0; j < width; ++j) {
        
                const itype dest = i*width+j;
        
                len[dest] = 0;
                p_i[dest] = NIL;
                p_j[dest] = NIL;
                pen[dest] = INFTY;
            }
        }
        
        #pragma omp barrier
    
        for (itype i = 0; i < Nsize-1; ++i) {
            # pragma omp for
            for (itype j = 0; j < width-1; ++j) {
            
                const itype src = i*width+j;
                const itype border_k = std::min(St0, Nsize-i-1);

                for (itype k = 0; k < border_k; ++k) {
                
                    const itype i1k = i+1+k, j1 = j+1, j1M = j+1+M->left;
                
                    ftype a0 = H->at(j1M)-N->at(i1k);
                    a0 = (1-E_k[k])*a_0[src]+E_k[k]*a0;
                
                    ftype dst = H->at(j1M)-a0-N->at(i1k);
                    
                    if (squared)
                        dst *= dst;
                    else
                        dst = ABS(dst);

                    const itype trg = i1k*width+j1;
                
                    if (pen[src]+dst < pen[trg]/len[trg]*(len[src]+1)) {
                    
                        len[trg] = len[src]+1;
                        p_i[trg] = i;
                        p_j[trg] = j;
                     
                        pen[trg] = pen[src]+dst;
                        a_0[trg] = a0;
                    }
                }
            
                const itype border_l = std::min(St1, width-j-1);

                for (itype l = 1; l < border_l; ++l) {

                    const itype i1 = i+1, j1l = j+1+l, j1lM = j+1+l+M->left;

                    ftype a0 = H->at(j1lM)-N->at(i1);
                    a0 = (1-E_l[l])*a_0[src]+E_l[l]*a0;

                    ftype dst = H->at(j1lM)-a0-N->at(i1);
                    
                    if (squared)
                        dst *= dst;
                    else
                        dst = ABS(dst);

                    const itype trg = (i1)*width+j1l;
                
                    if (pen[src]+dst < pen[trg]/len[trg]*(len[src]+1)) {
                    
                        len[trg] = len[src]+1;
                        p_i[trg] = i;
                        p_j[trg] = j;
                     
                        pen[trg] = pen[src]+dst;
                        a_0[trg] = a0;
                    } 
                }
            }
        }  
    } // end pragma omp parallel
  
    itype i = Nsize-1, j = width-1;
    
    while(i != 0 && j != 0) {
    
        itype src = i*width+j;
        
        L->push_back(a_0[src]);
        X->push_back(j+M->left);
        Y->push_back(N->at(i)+a_0[src]);
        
        i = p_i[src];
        j = p_j[src];
    }
    
    delete[] len;
    delete[] p_i;
    delete[] p_j;
    delete[] pen;
    delete[] a_0;
    
    delete[] E_k;
    delete[] E_l;

    return 0;
}


int backtrace(series *N, series *H, single_match *M, series *L, coords *X,
              series *Y, itype St0, itype St1, ftype E, bool omp, bool squared){
    if (omp) {
        if (squared)
            return meta_backtrace<true, true> (N, H, M, L, X, Y, St0, St1, E);
        else
            return meta_backtrace<true, false> (N, H, M, L, X, Y, St0, St1, E);
    } else {
        if (squared)
            return meta_backtrace<false, true> (N, H, M, L, X, Y, St0, St1, E);
        else
            return meta_backtrace<false, false> (N, H, M, L, X, Y, St0, St1, E);
    }
}

/*******************************************************************************
* main
*******************************************************************************/

int main(int argc, char **args) {

    series *N = new series();
    series *H = new series();
    result *R = new result();
   
    for (itype i = 0; i < 1000; ++i)
        N->push_back(sin(i/100.0));
    
    for (itype i = 0; i < 1000000; ++i)
        H->push_back(0);
        
    for (itype i = 701234; i < 702234; ++i)
        H->at(i)=sin(i/100.0);
   
    match(N, H, R, 2, 2, 0.01, true, true);
 
    std::cout << R->at(0).left << " " << R->at(0).right << " " << R->at(0).penalty << std::endl; 
 
    delete N;
    delete H;
    delete R;  
   
    return 0;
}
