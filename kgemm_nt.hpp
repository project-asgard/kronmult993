#ifndef KGEMM_NT_HPP
#define KGEMM_NT_HPP 1

#include "kroncommon.hpp"

//  -----------------------
//  NotransA and TransB case
//  C = alpha*A*transpose(B) + beta *C
//  -----------------------
template <typename T>
DEVICE_FUNCTION void kgemm_nt(int const mm, int const nn, int const kk,
                              T const alpha, T const *const A_, int const ldA,
                              T const *const B_, int const ldB, T const beta,
                              T *C_, int const ldC) {

#ifdef USE_GPU

  // ---------------------------
  // use matlab 1 based indexing
  // ---------------------------

  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  // -----------------------------------------
  // reorganize threads as nx_threads by ny_threads
  // -----------------------------------------

  int const ij_start = threadIdx.x;
  int const ij_size = blockDim.x;

#else

  int const ij_start = 0;
  int const ij_size = 1;

#endif

  //  ------------------------------------
  //  commonly  mm is large, but kk, nn are small
  //
  //  consider increasing nb_m for more effective
  //  use of shared cache
  //
  //  ------------------------------------

  auto A = [&](int const ia, int const ja) -> T const & {
    return (A_[indx2(ia, ja, ldA)]);
  };

  auto B = [&](int const ib, int const jb) -> T const & {
    return (B_[indx2(ib, jb, ldB)]);
  };

  auto C = [&](int const ic, int const jc) -> T & {
    return (C_[indx2(ic, jc, ldC)]);
  };

      // ---------------------------
      // perform matrix calculations
      // ---------------------------

      for (int ij0 = ij_start; ij0 < (mm * nn); ij0 += ij_size) {
        int const i = ij0 % mm;
        int const j = (ij0 - i) / mm;
        T cij = 0;

          for (int k = 0; k < kk; k++) {
            cij += A(i, k) * B(j, k);
          };

        
        // ------------------
        // store results to C
        // ------------------

        if (beta == 1) {
          atomicAdd(&(C(i, j)), alpha * cij);
        } else if (beta == 0) {
          C(i, j) = alpha * cij;
        } else {
          C(i, j) = beta * C(i, j) + alpha * cij;
        };
      };
}

#endif
