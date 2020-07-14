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


  int const ij_start = threadIdx.x + 1;
  int const ij_size = blockDim.x;

#else

  int const ix_start = 1;
  int const ix_size = 1;
  int const iy_start = 1;
  int const iy_size = 1;

  int const ij_start = 1;
  int const ij_size = 1;
#endif

  assert(ix_start >= 1);
  assert(iy_start >= 1);
  assert(ix_size >= 1);
  assert(iy_size >= 1);

  //  ------------------------------------
  //  commonly  mm is large, but kk, nn are small
  //
  //  consider increasing nb_m for more effective
  //  use of shared cache
  //
  //  ------------------------------------

  auto A = [&](int const ia, int const ja) -> T const & {
    return (A_[indx2f(ia, ja, ldA)]);
  };

  auto B = [&](int const ib, int const jb) -> T const & {
    return (B_[indx2f(ib, jb, ldB)]);
  };

  auto C = [&](int const ic, int const jc) -> T & {
    return (C_[indx2f(ic, jc, ldC)]);
  };

      // ---------------------------
      // perform matrix calculations
      // ---------------------------

      for (int ij0 = ij_start; ij0 < (mm * nn); ij0 += ij_size) {
        int const i = (ij0 % mm) + 1;
        int const j = (ij0 - (i - 1)) / mm + 1;
        T cij = 0;
        bool constexpr use_pointer = true;
        if (use_pointer) {
          int k = 1;

          T const *Ap = &(A(i, k));
          int64_t const inc_A = &(A(i, k + 1)) - Ap;
          T const *Bp = &(B(j, k));
          int64_t const inc_B = &(B(j, k + 1)) - Bp;
          for (k = 0; k < kk; k++) {
            cij += (*Ap) * (*Bp);
            Ap += inc_A;
            Bp += inc_B;
          };
        } else {
          for (int k = 1; k <= kk; k++) {
            cij += A(i, k) * B(j, k);
          };
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
