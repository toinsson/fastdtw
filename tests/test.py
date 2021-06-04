import fastdtw2
import numpy as np

X = np.random.random((1000, 9))
Y = np.random.random((1000, 9))


print("###: TEST 0")
D0 = fastdtw2.fastdtw(X, Y)
D1 = fastdtw2.fastdtw(X, Y, dist='mahalanobis_diag', weights_list=np.ones(9))
print(D0[0], D1[0])
print("D0 == D1", np.isclose(D0[0], D1[0]))

print("###: TEST 1")
D0 = fastdtw2.fastdtw(X, Y, dist=2)
D01 = fastdtw2.fastdtw(X, Y)
D1 = fastdtw2.fastdtw(X, Y, dist='mahalanobis_full', weights_list=np.eye(9).reshape(-1))
D2 = fastdtw2.fastdtw(X, Y, dist='mahalanobis_full', weights_list=np.zeros(81).reshape(-1))

print(D0[0], D01[0], D1[0], D2[0])

ic_random = np.cov(X.T)
def maha(x, y):
    # https://github.com/scipy/scipy/blob/v1.6.3/scipy/spatial/distance.py#L1049-L1093
    delta = x - y
    m = np.dot(np.dot(delta, ic_random), delta)
    return np.sqrt(m)


D0 = fastdtw2.fastdtw(X, Y, dist=maha)
D1 = fastdtw2.fastdtw(X, Y, dist='mahalanobis_full', weights_list=ic_random.reshape(-1))
print(D0[0], D1[0])
print("D0 == D1", np.isclose(D0[0], D1[0]))