import numpy as np
import time

n = 500

X = np.random.randn(n, 6 * n)
R = np.corrcoef(X)
print(R.shape)


def sub_matrices_product_quick(R):
    n_stocks = R.shape[0]
    R_inv = np.linalg.pinv(R)
    for i in range(n_stocks):
        idx = np.concatenate((np.arange(0, i), np.arange(i + 1, n_stocks))).astype(np.int32)
        U = np.zeros([n_stocks, 2])
        U[i, 0] = 1
        U[idx, 1] = R[idx, i]

        V = np.zeros([2, n_stocks])
        V[1, i] = 1
        V[0, idx] = R[i, idx]

        R_other_inv_SM = (R_inv -
                          np.dot(np.dot(R_inv,
                                        np.dot(-U,
                                               np.linalg.inv(np.eye(2) +
                                                             np.dot(V,
                                                                    np.dot(R_inv,
                                                                           -U))))),
                                 np.dot(V,
                                        R_inv)))
        R_other_inv_SM = R_other_inv_SM[idx, :][:, idx]


def sub_matrices_product(R):
    for i in range(R.shape[0]):
        R2 = np.delete(np.delete(R, i, axis=0), i, axis=1)
        np.linalg.pinv(R2)


# n_it = 10
# start = time.time()
# for k in range(n_it):
#     print(k)
#     sub_matrices_product_quick(R)
# end = time.time()
# time_perit_quick = (end - start) / n_it
# print("sub_matrices_product_quick: ", time_perit_quick)

n_it = 1
start = time.time()
for k in range(n_it):
    print(k)
    sub_matrices_product(R)
end = time.time()
time_perit = (end - start) / n_it
print("sub_matrices_product: ", time_perit)

# print("speed-up factor: ", time_perit / time_perit_quick)

if __name__ == "__main__":
    pass
