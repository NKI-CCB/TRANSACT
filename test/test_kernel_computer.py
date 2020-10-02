import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed

from transact.kernel_computer import KernelComputer
from transact.matrix_operations import _left_center_kernel, _center_kernel


n_source_samples = 30
n_target_samples = 50
n_dimensions = 400

@pytest.fixture(scope='module')
def source_data():
    return np.random.normal(0,1,size=(n_source_samples, n_dimensions))


@pytest.fixture(scope='module')
def target_data():
    return np.random.normal(0,1,size=(n_target_samples, n_dimensions))



class TestKernelComputer():

    @pytest.fixture(scope='class')
    def linear_kernel_computer(self):
        return KernelComputer('linear')

    @pytest.fixture(scope='class')
    def uncentered_linear_kernel_computer(self, source_data, target_data):
        return KernelComputer('linear').fit(source_data, target_data, center=False)

    @pytest.fixture(scope='class')
    def rbf_kernel_computer(self):
        return KernelComputer('rbf')


    @pytest.fixture(scope='class')
    def uncentered_rbf_kernel_computer(self, source_data, target_data):
        return KernelComputer('rbf').fit(source_data, target_data, center=False)


    def test_source_data_property_no_target(self,
                                        source_data,
                                        linear_kernel_computer):

        linear_kernel_computer.source_data = source_data

        assert linear_kernel_computer.kernel_matrix_ is None
        assert linear_kernel_computer.kernel_submatrices is None
        assert linear_kernel_computer.k_s is None
        assert linear_kernel_computer.k_t is None
        assert linear_kernel_computer.k_st is None
        assert linear_kernel_computer.k_ts is None

        linear_kernel_computer.source_data = None


    def test_source_data_property_with_target(self,
                                        source_data,
                                        target_data,
                                        linear_kernel_computer):

        linear_kernel_computer.target_data = target_data
        linear_kernel_computer.source_data = source_data

        assert linear_kernel_computer.kernel_matrix_ is not None
        assert linear_kernel_computer.kernel_submatrices is not None
        assert linear_kernel_computer.k_s is not None
        assert linear_kernel_computer.k_t is not None
        assert linear_kernel_computer.k_st is not None
        assert linear_kernel_computer.k_ts is not None

        linear_kernel_computer.target_data = None
        linear_kernel_computer.source_data = None


    def test_target_data_property_no_source(self,
                                        target_data,
                                        linear_kernel_computer):

        linear_kernel_computer.target_data = target_data

        assert linear_kernel_computer.kernel_matrix_ is None
        assert linear_kernel_computer.kernel_submatrices is None
        assert linear_kernel_computer.k_s is None
        assert linear_kernel_computer.k_t is None
        assert linear_kernel_computer.k_st is None
        assert linear_kernel_computer.k_ts is None

        linear_kernel_computer.target_data = None


    def test_target_data_property_with_source(self,
                                        source_data,
                                        target_data,
                                        linear_kernel_computer):

        linear_kernel_computer.source_data = source_data
        print(linear_kernel_computer._source_data)
        linear_kernel_computer.target_data = target_data

        assert linear_kernel_computer.kernel_matrix_ is not None
        assert linear_kernel_computer.kernel_submatrices is not None
        assert linear_kernel_computer.k_s is not None
        assert linear_kernel_computer.k_t is not None
        assert linear_kernel_computer.k_st is not None
        assert linear_kernel_computer.k_ts is not None

        linear_kernel_computer.target_data = None
        linear_kernel_computer.source_data = None


    def test_linear_fit(self,
                    source_data,
                    target_data,
                    linear_kernel_computer):

        linear_kernel_computer.fit(source_data, target_data)

        assert linear_kernel_computer.kernel_matrix_.shape == (n_source_samples+n_target_samples, n_source_samples+n_target_samples)
        assert len(linear_kernel_computer.kernel_submatrices.keys()) == 4
        assert 'source' in linear_kernel_computer.kernel_submatrices.keys()
        assert 'target' in linear_kernel_computer.kernel_submatrices.keys()
        assert 'source-target' in linear_kernel_computer.kernel_submatrices.keys()
        assert 'target-source' in linear_kernel_computer.kernel_submatrices.keys()
        assert linear_kernel_computer.k_s.shape == (n_source_samples, n_source_samples)
        assert linear_kernel_computer.k_t.shape == (n_target_samples, n_target_samples)
        assert linear_kernel_computer.k_st.shape == (n_source_samples, n_target_samples)
        assert linear_kernel_computer.k_ts.shape == (n_target_samples, n_source_samples)

    def test_rbf_fit(self,
                    source_data,
                    target_data,
                    rbf_kernel_computer):

        rbf_kernel_computer.fit(source_data, target_data)

        assert rbf_kernel_computer.kernel_matrix_.shape == (n_source_samples+n_target_samples, n_source_samples+n_target_samples)
        assert len(rbf_kernel_computer.kernel_submatrices.keys()) == 4
        assert 'source' in rbf_kernel_computer.kernel_submatrices.keys()
        assert 'target' in rbf_kernel_computer.kernel_submatrices.keys()
        assert 'source-target' in rbf_kernel_computer.kernel_submatrices.keys()
        assert 'target-source' in rbf_kernel_computer.kernel_submatrices.keys()
        assert rbf_kernel_computer.k_s.shape == (n_source_samples, n_source_samples)
        assert rbf_kernel_computer.k_t.shape == (n_target_samples, n_target_samples)
        assert rbf_kernel_computer.k_st.shape == (n_source_samples, n_target_samples)
        assert rbf_kernel_computer.k_ts.shape == (n_target_samples, n_source_samples)


    def test_center_linear(self,
                        source_data, 
                        target_data, 
                        linear_kernel_computer, 
                        uncentered_linear_kernel_computer):

        linear_kernel_computer.fit(source_data, target_data, center=True)

        assert np.any(np.not_equal(linear_kernel_computer.k_s,
                                uncentered_linear_kernel_computer.k_s))
        assert np.any(np.not_equal(linear_kernel_computer.k_t,
                                uncentered_linear_kernel_computer.k_t))
        assert np.any(np.not_equal(linear_kernel_computer.k_st,
                                uncentered_linear_kernel_computer.k_st))

        np.testing.assert_almost_equal(_center_kernel(uncentered_linear_kernel_computer.k_s),
                                    linear_kernel_computer.k_s)
        np.testing.assert_almost_equal(_center_kernel(uncentered_linear_kernel_computer.k_t),
                                    linear_kernel_computer.k_t)
        np.testing.assert_almost_equal(_center_kernel(uncentered_linear_kernel_computer.k_st),
                                    linear_kernel_computer.k_st)


    def test_center_rbf(self,
                        source_data, 
                        target_data, 
                        rbf_kernel_computer, 
                        uncentered_rbf_kernel_computer):

        rbf_kernel_computer.fit(source_data, target_data, center=True)

        assert np.any(np.not_equal(rbf_kernel_computer.k_s,
                                uncentered_rbf_kernel_computer.k_s))
        assert np.any(np.not_equal(rbf_kernel_computer.k_t,
                                uncentered_rbf_kernel_computer.k_t))
        assert np.any(np.not_equal(rbf_kernel_computer.k_st,
                                uncentered_rbf_kernel_computer.k_st))

        np.testing.assert_almost_equal(_center_kernel(uncentered_rbf_kernel_computer.k_s),
                                    rbf_kernel_computer.k_s)
        np.testing.assert_almost_equal(_center_kernel(uncentered_rbf_kernel_computer.k_t),
                                    rbf_kernel_computer.k_t)
        np.testing.assert_almost_equal(_center_kernel(uncentered_rbf_kernel_computer.k_st),
                                    rbf_kernel_computer.k_st)

    def test_transform(self,
                    source_data,
                    target_data,
                    rbf_kernel_computer):

        rbf_kernel_computer.fit(source_data, target_data)

        n_source_samples = source_data.shape[0]
        x_target_proj = rbf_kernel_computer.transform(target_data, center=True)
        np.testing.assert_almost_equal(x_target_proj[:,n_source_samples:], rbf_kernel_computer.k_t)
        np.testing.assert_almost_equal(x_target_proj[:,:n_source_samples], rbf_kernel_computer.k_ts)

        n_source_samples = source_data.shape[0]
        x_source_proj = rbf_kernel_computer.transform(source_data, center=True)
        np.testing.assert_almost_equal(x_source_proj[:,n_source_samples:], rbf_kernel_computer.k_st)
        np.testing.assert_almost_equal(x_source_proj[:,:n_source_samples], rbf_kernel_computer.k_s)
