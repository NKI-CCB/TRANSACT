import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from sklearn.datasets import make_spd_matrix
from sklearn.decomposition import PCA

from transact.interpolation import Interpolation
from transact.matrix_operations import _left_center_kernel
from transact.pv_computation import PVComputation
from transact.kernel_computer import KernelComputer


n_source_samples = 150
n_target_samples = 100
n_dimensions = 200
n_iter_spd_cov = 50
cov_sim_ratio = 0.9
n_jobs = 10
n_pc = 20
n_pv = 15
n_tau_comparison = 10
rbf_params = {
    'gamma':1/n_dimensions
}

def generate_covariance_matrix():
    spd_matrices = Parallel(n_jobs=n_jobs, verbose=1)(delayed(make_spd_matrix)(n_dimensions) for _ in range(n_iter_spd_cov))
    return np.mean(spd_matrices, axis=0)


@pytest.fixture(scope='module')
def common_cov_matrix():
    return generate_covariance_matrix()


@pytest.fixture(scope='module')
def source_cov_matrix(common_cov_matrix):
    return cov_sim_ratio * common_cov_matrix + (1-cov_sim_ratio)*generate_covariance_matrix()


@pytest.fixture(scope='module')
def target_cov_matrix(common_cov_matrix):
    return cov_sim_ratio * common_cov_matrix + (1-cov_sim_ratio)*generate_covariance_matrix()


@pytest.fixture(scope='module')
def source_data(source_cov_matrix):
    X = np.random.multivariate_normal(mean=np.zeros(n_dimensions),
                                     cov=source_cov_matrix,
                                     size=n_source_samples)
    return X


@pytest.fixture(scope='module')
def source_data_centered(source_data):
    return source_data - np.mean(source_data, 0)


@pytest.fixture(scope='module')
def target_data(target_cov_matrix):
    X = np.random.multivariate_normal(mean=np.zeros(n_dimensions),
                                     cov=target_cov_matrix,
                                     size=n_target_samples)
    return X


@pytest.fixture(scope='module')
def target_data_centered(target_data):
    return target_data - np.mean(target_data, 0)


@pytest.fixture(scope='module')
def source_PCA(source_data):
    return PCA(n_pc).fit(source_data)


@pytest.fixture(scope='module')
def target_PCA(target_data):
    return PCA(n_pc).fit(target_data)


@pytest.fixture(scope='module')
def source_PC_projected(source_data):
    return PCA(n_pc).fit_transform(source_data)


@pytest.fixture(scope='module')
def target_PC_projected(target_data):
    return PCA(n_pc).fit_transform(target_data)


@pytest.fixture(scope='module')
def linear_PV(source_PCA, target_PCA):
    M = source_PCA.components_.dot(target_PCA.components_.T)
    u,s,v = np.linalg.svd(M)
    return u.T.dot(source_PCA.components_), v.dot(target_PCA.components_)


@pytest.fixture(scope='module')
def source_linear_PV_projected(linear_PV, source_data):
    return source_data.dot(linear_PV[0].T)[:,:n_pv], source_data.dot(linear_PV[1].T)[:,:n_pv]


@pytest.fixture(scope='module')
def target_linear_PV_projected(linear_PV, target_data):
    return target_data.dot(linear_PV[0].T)[:,:n_pv], target_data.dot(linear_PV[1].T)[:,:n_pv]

@pytest.fixture(scope='module')
def rbf_principal_vectors(source_data, target_data):
    return PVComputation('rbf', rbf_params, n_components=n_pc, n_pv=n_pv).fit(source_data, target_data)

@pytest.fixture(scope='module')
def target_rbf_PV_projected(rbf_principal_vectors, target_data):
    return rbf_principal_vectors.transform(target_data)

@pytest.fixture(scope='module')
def source_rbf_PV_projected(rbf_principal_vectors, source_data):
    return rbf_principal_vectors.transform(source_data)

@pytest.fixture(scope='module')
def rbf_kernel_matrix(source_data, target_data):
    k = KernelComputer('rbf', rbf_params)
    k.fit(source_data, target_data, center=False)

    return k

@pytest.fixture(scope='module')
def linear_principal_vectors(source_data, target_data):
    return PVComputation('linear', n_components=n_pc, n_pv=n_pv).fit(source_data, target_data)

@pytest.fixture(scope='module')
def linear_kernel_matrix(source_data, target_data):
    k = KernelComputer('linear')
    k.fit(source_data, target_data, center=False)

    return k

class TestPVComputation():

    @pytest.fixture(scope='class')
    def interpolation(self):
        return Interpolation('linear')

    @pytest.fixture(scope='class')
    def interpolation_rbf(self, rbf_principal_vectors, rbf_kernel_matrix):
        clf_interpol = Interpolation('rbf', kernel_params=rbf_params)
        return clf_interpol.fit(rbf_principal_vectors, rbf_kernel_matrix, n_pv=n_pv)


    @pytest.fixture(scope='class')
    def interpolation_linear(self,
                                    linear_principal_vectors,
                                    linear_kernel_matrix):
        clf = Interpolation('linear')
        return clf.fit(linear_principal_vectors, linear_kernel_matrix, n_pv=n_pv)


    def test_linear_zero_tau(self,
                            interpolation_linear,
                            source_linear_PV_projected,
                            target_linear_PV_projected):

        X = interpolation_linear.project_data(0, source_only=True, target_only=False, center=False)
        np.testing.assert_almost_equal(X, source_linear_PV_projected[0])

        X = interpolation_linear.project_data(0, source_only=False, target_only=True, center=False)
        np.testing.assert_almost_equal(X, target_linear_PV_projected[0])

    def test_linear_one_tau(self,
                            interpolation_linear,
                            source_linear_PV_projected,
                            target_linear_PV_projected):

        X = interpolation_linear.project_data(1, source_only=True, target_only=False, center=False)
        np.testing.assert_almost_equal(X, source_linear_PV_projected[1])

        X = interpolation_linear.project_data(1, source_only=False, target_only=True, center=False)
        np.testing.assert_almost_equal(X, target_linear_PV_projected[1])


    def test_rbf_zero_tau(self,
                        interpolation_rbf,
                        source_rbf_PV_projected,
                        target_rbf_PV_projected):

        X = interpolation_rbf.project_data(0, source_only=True, target_only=False, center=False)
        np.testing.assert_almost_equal(X, source_rbf_PV_projected['source'])

        X = interpolation_rbf.project_data(0, source_only=False, target_only=True, center=False)
        np.testing.assert_almost_equal(X, target_rbf_PV_projected['source'])


    def test_rbf_one_tau(self,
                        interpolation_rbf,
                        source_rbf_PV_projected,
                        target_rbf_PV_projected):

        X = interpolation_rbf.project_data(1, source_only=True, target_only=False, center=False)
        np.testing.assert_almost_equal(X, source_rbf_PV_projected['target'])

        X = interpolation_rbf.project_data(1, source_only=False, target_only=True, center=False)
        np.testing.assert_almost_equal(X, target_rbf_PV_projected['target'])
        

    def test_transform_linear(self,
                        interpolation_linear,
                        source_data,
                        target_data):
        # Test that transform and project_data gives same result for source and target data
        # Here specific to linear scenario.
        for _ in range(n_tau_comparison):
            tau = np.random.uniform(size=n_pv)
            # Without centering
            np.testing.assert_almost_equal(interpolation_linear.project_data(tau, source_only=True, target_only=False, center=False),
                                        interpolation_linear.transform(source_data, tau=tau, center=False))
            np.testing.assert_almost_equal(interpolation_linear.project_data(tau, source_only=False, target_only=True, center=False),
                                        interpolation_linear.transform(target_data, tau=tau, center=False))

            # With centering
            np.testing.assert_almost_equal(interpolation_linear.project_data(tau, source_only=True, target_only=False, center=True),
                                        interpolation_linear.transform(source_data, tau=tau, center=True))
            np.testing.assert_almost_equal(interpolation_linear.project_data(tau, source_only=False, target_only=True, center=True),
                                        interpolation_linear.transform(target_data, tau=tau, center=True))


    def test_transform_rbf(self,
                        interpolation_rbf,
                        source_data,
                        target_data):
        # Test that transform and project_data gives same result for source and target data
        # Here specific to linear scenario.
        for _ in range(n_tau_comparison):
            tau = np.random.uniform(size=n_pv)
            # Without centering
            np.testing.assert_almost_equal(interpolation_rbf.project_data(tau, source_only=True, target_only=False, center=False),
                                        interpolation_rbf.transform(source_data, tau=tau, center=False))
            np.testing.assert_almost_equal(interpolation_rbf.project_data(tau, source_only=False, target_only=True, center=False),
                                        interpolation_rbf.transform(target_data, tau=tau, center=False))

            # With centering
            np.testing.assert_almost_equal(interpolation_rbf.project_data(tau, source_only=True, target_only=False, center=True),
                                        interpolation_rbf.transform(source_data, tau=tau, center=True))
            np.testing.assert_almost_equal(interpolation_rbf.project_data(tau, source_only=False, target_only=True, center=True),
                                        interpolation_rbf.transform(target_data, tau=tau, center=True))
