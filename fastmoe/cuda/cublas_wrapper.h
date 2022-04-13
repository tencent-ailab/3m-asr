#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H
#include <cublas_v2.h>
#include <c10/util/Half.h>

inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *Aarray[], int lda,
                                  const float           *Barray[], int ldb,
                                  const float           *beta,
                                  float           *Carray[], int ldc,
                                  int batchCount) {
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray,
                            lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double           *alpha,
                                  const double           *Aarray[], int lda,
                                  const double           *Barray[], int ldb,
                                  const double           *beta,
                                  double           *Carray[], int ldc,
                                  int batchCount) {
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                              ldb, beta, Carray, ldc, batchCount);
}

inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const __half           *alpha,
                                  const __half           *Aarray[], int lda,
                                  const __half           *Barray[], int ldb,
                                  const __half           *beta,
                                  __half           *Carray[], int ldc,
                                  int batchCount) {
    return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray,
                              lda, Barray, ldb, beta, Carray, ldc, batchCount);
}


inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const float           *alpha,
                                const float           *A, int lda,
                                const float           *B, int ldb,
                                const float           *beta,
                                float           *C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const double          *alpha,
                                const double          *A, int lda,
                                const double          *B, int ldb,
                                const double          *beta,
                                double          *C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const __half *alpha,
                                const __half *A, int lda,
                                const __half *B, int ldb,
                                const __half *beta,
                                __half *C, int ldc) {
    return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const c10::Half *alpha,
                                const c10::Half *A, int lda,
                                const c10::Half *B, int ldb,
                                const c10::Half *beta,
                                c10::Half *C, int ldc) {
    return cublasHgemm(handle, transa, transb, m, n, k,
            (const __half*)alpha,
            (const __half*)A, lda,
            (const __half*)B, ldb,
            (const __half*)beta,
            reinterpret_cast<__half*>(C), ldc);
}
#endif  // CUBLAS_WRAPPER_H