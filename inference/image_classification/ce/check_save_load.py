import nose.tools as tools
import numpy as np


class TestSaveLoadAPI(object):
    """
    Test Save Load inference model
    """

    def __init__(self):
        """
        __init__
        """
        pass

    def test_results(self):
        """
        compare two results
        """
        expect = np.loadtxt('./results_save_model.txt')
        result = np.loadtxt('./results_load_model.txt')
        assert len(expect) == len(result)
        for i in range(0, len(expect)):
            tools.assert_almost_equal(expect[i], result[i], delta=1e-4)
