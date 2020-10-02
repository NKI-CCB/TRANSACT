import numpy as np
import pytest
from sklearn.datasets import make_spd_matrix

from transact.matrix_operations import _sqrt_matrix
from transact.matrix_operations import _center_kernel
from transact.matrix_operations import _right_center_kernel
from transact.matrix_operations import _left_center_kernel
from transact.matrix_operations import centering_matrix

size = 20

@pytest.fixture(scope='module')
def random_matrix():
    return make_spd_matrix(n_dim=size)

@pytest.fixture(scope='module')
def C_n():
    return centering_matrix(size) 

class TestPVComputation():

    @pytest.fixture(scope='class')
    def sqrt_matrix(self, random_matrix):
        return _sqrt_matrix(random_matrix)

    def test_sqrt_shape(self, sqrt_matrix, random_matrix):
        assert sqrt_matrix.shape == random_matrix.shape

    def test_sqrt_reconstruction(self, sqrt_matrix, random_matrix):
        np.testing.assert_almost_equal(sqrt_matrix.dot(sqrt_matrix),
                                    random_matrix)


    def test_commutativity_centering(self, random_matrix):
        x = _left_center_kernel(random_matrix)
        x = _right_center_kernel(x)
        np.testing.assert_almost_equal(x, _center_kernel(random_matrix)) 

        x = _right_center_kernel(random_matrix)
        x = _left_center_kernel(x)
        np.testing.assert_almost_equal(x, _center_kernel(random_matrix)) 

    def test_properties_centering_matrix(self, C_n):
        # Idempotent
        np.testing.assert_almost_equal(C_n, C_n.dot(C_n))

        # Spectrum is [1]*(n-1) + [0]
        spectrum = np.ones(size)
        spectrum[0] = 0
        np.testing.assert_almost_equal(np.linalg.eigvalsh(C_n), spectrum)

        # Null space is ones
        np.testing.assert_almost_equal(C_n.dot(np.ones(size)), np.zeros(size))



