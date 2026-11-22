import unittest
import numpy as np

from team7_quant.module5 import lw_constant_correlation, black_litterman_posterior, optimize_long_only_capped


class Module5Tests(unittest.TestCase):
    def test_lw_golden(self):
        r=.01*np.array([[-2,-1,1],[1,-2,1],[-1,2,-3],[0,-1,1],[0,0,-1],[1,2,2],[-3,2,-1],[2,0,1]],float)
        out=lw_constant_correlation(r)
        expected=np.array([[.00027857142857142857,-.000049789147753238289,.000049900630053572819],
                           [-.000049789147753238289,.00025,-.000068263807025821480],
                           [.000049900630053572819,-.000068263807025821480,.00026964285714285714]])
        np.testing.assert_allclose(out.covariance_daily,expected,atol=1e-14,rtol=0)
        self.assertAlmostEqual(out.rbar,-.0898388706308174665,12)
        self.assertAlmostEqual(out.shrinkage,.5246208778011419394,12)

    def test_bl_analytic(self):
        n=30; sigma=(1/25)*np.eye(n)+(17/750)*np.ones((n,n)); b=np.full(n,1/n)
        mu,d=black_litterman_posterior(sigma,b,list(range(5)))
        np.testing.assert_allclose(d["pi"],.06,atol=1e-12)
        np.testing.assert_allclose(mu[:5],.075,atol=1e-12)
        np.testing.assert_allclose(mu[5:],.07108695652173913,atol=1e-12)
        w,diag=optimize_long_only_capped(mu,sigma,2.5,.1)
        np.testing.assert_allclose(w[:5],.06594202898550724,atol=2e-6)
        np.testing.assert_allclose(w[5:],.02681159420289855,atol=2e-6)
        self.assertAlmostEqual(w.sum(),1,8)


if __name__ == "__main__": unittest.main()

