import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from transact.pv_computation import PVComputation


n_source_samples = 30
n_target_samples = 50
n_dimensions = 100
n_iter_spd_cov = 20
cov_sim_ratio = 0.7
n_jobs = 10
n_pc = 15
n_pv = 10
rbf_param = {
    'gamma': 1 / n_dimensions
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
    X = np.random.multivariate_normal(mean=np.ones(n_dimensions),
                                     cov=source_cov_matrix,
                                     size=n_source_samples)
    return X - np.mean(X, axis=0)


@pytest.fixture(scope='module')
def target_data(target_cov_matrix):
    X = np.random.multivariate_normal(mean=np.zeros(n_dimensions),
                                     cov=target_cov_matrix,
                                     size=n_target_samples)
    return X - np.mean(X, axis=0)


@pytest.fixture(scope='module')
def source_offset():
    return np.random.randint(-100, 100, size=n_dimensions)

@pytest.fixture(scope='module')
def target_offset():
    return np.random.randint(-100, 100, size=n_dimensions)


@pytest.fixture(scope='module')
def source_offset_data(source_cov_matrix, source_offset):
    X = np.random.multivariate_normal(mean=source_offset,
                                     cov=source_cov_matrix,
                                     size=n_source_samples)
    return X


@pytest.fixture(scope='module')
def target_offset_data(target_cov_matrix, target_offset):
    X = np.random.multivariate_normal(mean=target_offset,
                                     cov=target_cov_matrix,
                                     size=n_target_samples)
    return X


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
def linear_canonical_angle(source_PCA, target_PCA):
    M = source_PCA.components_.dot(target_PCA.components_.T)
    u,s,v = np.linalg.svd(M)
    return s[:n_pv]

@pytest.fixture(scope='module')
def source_linear_PV_projected(linear_PV, source_data):
    return source_data.dot(linear_PV[0].T)[:,:n_pv], source_data.dot(linear_PV[1].T)[:,:n_pv]


@pytest.fixture(scope='module')
def target_linear_PV_projected(linear_PV, target_data):
    return target_data.dot(linear_PV[0].T)[:,:n_pv], target_data.dot(linear_PV[1].T)[:,:n_pv]



class TestPVComputation():


    @pytest.fixture(scope='class')
    def kernel_pv(self,
                source_data,
                target_data):
        clf = PVComputation('linear', n_components=n_pc, n_pv=n_pv)
        return clf.fit(source_data, target_data, 'two-stage')


    @pytest.fixture(scope='class')
    def kernel_pv_offset(self,
                    source_offset_data,
                    target_offset_data):
        clf_offset = PVComputation('linear', n_components=n_pc, n_pv=n_pv)
        return clf_offset.fit(source_offset_data, target_offset_data, 'two-stage')


    @pytest.fixture(scope='class')
    def kernel_pv_offset_centered(self,
                                source_offset_data,
                                target_offset_data):
        """
        Create same classifier than kernel_pv_offset, but first center the data.
        The idea is to see the influence of kernel centering in the linear setting -- which
        must be completly equivalent.
        """
        clf_offset_centered = PVComputation('linear', n_components=n_pc, n_pv=n_pv)
        x_source = StandardScaler(with_mean=True, with_std=False).fit_transform(source_offset_data)
        x_target = StandardScaler(with_mean=True, with_std=False).fit_transform(target_offset_data)
        return clf_offset_centered.fit(x_source, x_target, 'two-stage')


    @pytest.fixture(scope='class')
    def kernel_pv_source_projected(self,
                                source_data,
                                kernel_pv):
        return kernel_pv.transform(source_data)


    @pytest.fixture(scope='class')
    def kernel_pv_target_projected(self,
                                target_data,
                                kernel_pv):
        return kernel_pv.transform(target_data)

    
    @pytest.fixture(scope='class')
    def kernel_pv_source_offset_projected(self,
                                        source_offset_data,
                                        kernel_pv_offset):
        return kernel_pv_offset.transform(source_offset_data)


    @pytest.fixture(scope='class')
    def kernel_pv_target_offset_projected(self,
                                        target_offset_data,
                                        kernel_pv_offset):
        return kernel_pv_offset.transform(target_offset_data)


    @pytest.fixture(scope='class')
    def kernel_pv_source_offset_centered_projected(self,
                                                source_offset_data,
                                                kernel_pv_offset_centered):
        """
        We project the non-centered data to get a good comparison between the two.
        """
        return kernel_pv_offset_centered.transform(source_offset_data)


    @pytest.fixture(scope='class')
    def kernel_pv_target_offset_centered_projected(self,
                                                target_offset_data,
                                                kernel_pv_offset_centered):
        """
        We project the non-centered data to get a good comparison between the two.
        """
        return kernel_pv_offset_centered.transform(target_offset_data)


    def test_data_dim(self, source_data, target_data):
        assert source_data.shape == (n_source_samples, n_dimensions)
        assert target_data.shape == (n_target_samples, n_dimensions)


    def test_coefficients_shape(self, kernel_pv, kernel_pv_offset, kernel_pv_offset_centered):
        assert kernel_pv.gamma_coef['source'].shape == (n_pv, n_source_samples)
        assert kernel_pv.gamma_coef['target'].shape == (n_pv, n_target_samples)

        assert kernel_pv_offset.gamma_coef['source'].shape == (n_pv, n_source_samples)
        assert kernel_pv_offset.gamma_coef['target'].shape == (n_pv, n_target_samples)

        assert kernel_pv_offset_centered.gamma_coef['source'].shape == (n_pv, n_source_samples)
        assert kernel_pv_offset_centered.gamma_coef['target'].shape == (n_pv, n_target_samples)


    def test_dim_reduction(self,
                        source_data,
                        target_data,
                        kernel_pv,
                        source_PC_projected,
                        target_PC_projected):

        source_data_proj = kernel_pv.dim_reduc_clf_['source'].transform(source_data)
        target_data_proj = kernel_pv.dim_reduc_clf_['target'].transform(target_data)

        np.testing.assert_almost_equal(source_data_proj, source_PC_projected)
        np.testing.assert_almost_equal(target_data_proj, target_PC_projected)


    def test_source_projection(self,
                            source_linear_PV_projected,
                            kernel_pv_source_projected):
        np.testing.assert_almost_equal(source_linear_PV_projected[0],
                                    kernel_pv_source_projected['source'])
        np.testing.assert_almost_equal(source_linear_PV_projected[1],
                                    kernel_pv_source_projected['target'])


    def test_target_projection(self,
                            target_linear_PV_projected,
                            kernel_pv_target_projected):
        np.testing.assert_almost_equal(target_linear_PV_projected[0],
                                    kernel_pv_target_projected['source'])
        np.testing.assert_almost_equal(target_linear_PV_projected[1],
                                    kernel_pv_target_projected['target'])


    def test_canonical_angle_value(self,
                                kernel_pv,
                                linear_canonical_angle):
        angles = kernel_pv.canonical_angles
        np.testing.assert_equal(np.sort(angles), angles)
        np.testing.assert_almost_equal(angles, np.arccos(linear_canonical_angle))


    def test_linear_dim_reduction_offset_parameters(self,
                        kernel_pv_offset,
                        kernel_pv_offset_centered):

        np.testing.assert_almost_equal(kernel_pv_offset.dim_reduc_clf_['source'].alphas_,
                                    kernel_pv_offset_centered.dim_reduc_clf_['source'].alphas_)
        np.testing.assert_almost_equal(kernel_pv_offset.dim_reduc_clf_['target'].alphas_,
                                    kernel_pv_offset_centered.dim_reduc_clf_['target'].alphas_)

        np.testing.assert_almost_equal(kernel_pv_offset.dim_reduc_clf_['source'].lambdas_,
                                    kernel_pv_offset_centered.dim_reduc_clf_['source'].lambdas_)
        np.testing.assert_almost_equal(kernel_pv_offset.dim_reduc_clf_['target'].lambdas_,
                                    kernel_pv_offset_centered.dim_reduc_clf_['target'].lambdas_)


    def test_linear_pv_offset_parameters(self,
                            kernel_pv_offset,
                            kernel_pv_offset_centered):

        np.testing.assert_almost_equal(kernel_pv_offset.beta_coef['source'],
                                    kernel_pv_offset_centered.beta_coef['source'])
        np.testing.assert_almost_equal(kernel_pv_offset.beta_coef['target'],
                                    kernel_pv_offset_centered.beta_coef['target'])

        np.testing.assert_almost_equal(kernel_pv_offset.gamma_coef['source'],
                                    kernel_pv_offset_centered.gamma_coef['source'])
        np.testing.assert_almost_equal(kernel_pv_offset.gamma_coef['target'],
                                    kernel_pv_offset_centered.gamma_coef['target'])


    def test_linear_kernel_centering(self, kernel_pv_offset, kernel_pv_offset_centered):
        """
        Test the influence of kernel centering using linear kernel. Centering before of after
        is theoretically equivalent
        """
        np.testing.assert_almost_equal(kernel_pv_offset.kernel_values_.k_s, kernel_pv_offset_centered.kernel_values_.k_s)
        np.testing.assert_almost_equal(kernel_pv_offset.kernel_values_.k_t, kernel_pv_offset_centered.kernel_values_.k_t)
        np.testing.assert_almost_equal(kernel_pv_offset.kernel_values_.k_st, kernel_pv_offset_centered.kernel_values_.k_st)


    def test_linear_source_projection_with_offset(self,
                                            kernel_pv_source_offset_projected,
                                            kernel_pv_source_offset_centered_projected):

        np.testing.assert_almost_equal(kernel_pv_source_offset_centered_projected['source'],
                                    kernel_pv_source_offset_projected['source'])
        np.testing.assert_almost_equal(kernel_pv_source_offset_centered_projected['target'],
                                    kernel_pv_source_offset_projected['target'])


    def test_linear_target_projection_with_offset(self,
                                            kernel_pv_target_offset_projected,
                                            kernel_pv_target_offset_centered_projected):

        np.testing.assert_almost_equal(kernel_pv_target_offset_centered_projected['source'],
                                    kernel_pv_target_offset_projected['source'])
        np.testing.assert_almost_equal(kernel_pv_target_offset_centered_projected['target'],
                                    kernel_pv_target_offset_projected['target'])

    