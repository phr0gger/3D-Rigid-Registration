import numpy as np

a = np.array([[0.5449, 0.1955, 0.9227], [0.6862, 0.7202, 0.8004], [0.8936, 0.7218, 0.2859],
              [0.0548, 0.8778, 0.5437], [0.3037, 0.5824, 0.9848], [0.0462, 0.0707, 0.7157]])
b = np.array([[2.5144, 7.0691, 1.9754], [2.8292, 7.4454, 2.2224], [3.3518, 7.3060, 2.1198],
              [2.8392, 7.8455, 1.6229], [2.4901, 7.5449, 1.9518], [2.4273, 7.1354, 1.4349]])

a1 = np.column_stack((a,np.ones(a.shape[0])))
b1 = np.column_stack((b,np.ones(b.shape[0])))

matrixA = np.tensordot(a1,a1.T,axes=[0,1])
matrixR = np.tensordot(a1,b1.T,axes=[0,1])
matrixAinv = np.linalg.inv(matrixA)

tm = np.tensordot(matrixAinv,matrixR,axes=1).transpose()
# L2 norm is np.linalg.norm()
errors = b1 - [tm @ x for x in a1]
mean_error = np.mean([np.linalg.norm(y) for y in errors])  # 0.0109002
sum_error = np.sum([np.linalg.norm(y) for y in errors])  # 0.0654013

# print('\nA (23)\n',np.array2string(matrixA, precision=8))
# print('\n(20-22)\n',np.array2string(matrixR, precision=8))
print('\nT (24)\n',np.array2string(tm, precision=4, suppress_small=True))
print(f'Mean Error: {mean_error:.3f}')
print(f'Sum Error:  {sum_error:.3f}')
print(f'Froebenius: {np.linalg.norm(errors):.3f}')
