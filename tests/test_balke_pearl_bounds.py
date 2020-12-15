import pandas as pd
import numpy as np

from bpbounds.balke_pearl_bounds import ace_balke_pearl_bounds, print_results


def test_ace_balke_pearl_bounds_bivariate():
    x = [1, 0, 1, 0, 1]
    y = [1, 1, 0, 0, 1]
    z = [1, 0, 1, 0, 1]
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    res = ace_balke_pearl_bounds(df['x'], df['y'], df['z'])
    assert res['nzcats'] == 2
    assert res['inequality']
    np.testing.assert_almost_equal(res['bplb'], 0.1666, 4)
    np.testing.assert_almost_equal(res['bpub'], 0.1666, 4)
    np.testing.assert_almost_equal(res['p11low'], 0.6666, 4)
    np.testing.assert_almost_equal(res['p11upp'], 0.6666, 4)
    np.testing.assert_almost_equal(res['p10low'], 0.5, 4)
    np.testing.assert_almost_equal(res['p10upp'], 0.5, 4)
    np.testing.assert_almost_equal(res['crrlb'], 1.3333, 4)
    np.testing.assert_almost_equal(res['crrub'], 1.3333, 4)
    assert res['monoinequality']
    np.testing.assert_almost_equal(res['monobplb'], 0.1666, 4)
    np.testing.assert_almost_equal(res['monobpub'], 0.1666, 4)
    np.testing.assert_almost_equal(res['monop11low'], 0.6666, 4)
    np.testing.assert_almost_equal(res['monop11upp'], 0.6666, 4)
    np.testing.assert_almost_equal(res['monop10low'], 0.5, 4)
    np.testing.assert_almost_equal(res['monop10upp'], 0.5, 4)
    np.testing.assert_almost_equal(res['monocrrlb'], 1.3333, 4)
    np.testing.assert_almost_equal(res['monocrrub'], 1.3333, 4)


def test_ace_balke_pearl_bounds_dataframe():
    test_df = pd.concat([pd.DataFrame([[0, 0, 0]])] * 74, ignore_index=True).append(
        pd.concat([pd.DataFrame([[0, 0, 1]])] * 11514, ignore_index=True).append(
            pd.concat([pd.DataFrame([[1, 0, 0]])] * 34, ignore_index=True).append(
                pd.concat([pd.DataFrame([[1, 0, 1]])] * 2385, ignore_index=True).append(
                    pd.concat([pd.DataFrame([[1, 1, 0]])] * 12, ignore_index=True).append(
                        pd.concat([pd.DataFrame([[1, 1, 1]])] * 9663, ignore_index=True))))))
    test_df.columns = ['z', 'x', 'y']
    res = ace_balke_pearl_bounds(test_df['x'], test_df['y'], test_df['z'])
    assert res['nzcats'] == 2
    assert res['inequality']
    np.testing.assert_almost_equal(res['bplb'], -0.1946, 4)
    np.testing.assert_almost_equal(res['bpub'], 0.0053, 4)
    np.testing.assert_almost_equal(res['p11low'], 0.7989, 4)
    np.testing.assert_almost_equal(res['p11upp'], 0.9990, 4)
    np.testing.assert_almost_equal(res['p10low'], 0.9936, 4)
    np.testing.assert_almost_equal(res['p10upp'], 0.9936, 4)
    np.testing.assert_almost_equal(res['crrlb'], 0.8041, 4)
    np.testing.assert_almost_equal(res['crrub'], 1.0054, 4)
    assert res['monoinequality']
    np.testing.assert_almost_equal(res['monobplb'], -0.1946, 4)
    np.testing.assert_almost_equal(res['monobpub'], 0.0053, 4)
    np.testing.assert_almost_equal(res['monop11low'], 0.7989, 4)
    np.testing.assert_almost_equal(res['monop11upp'], 0.9990, 4)
    np.testing.assert_almost_equal(res['monop10low'], 0.9936, 4)
    np.testing.assert_almost_equal(res['monop10upp'], 0.9936, 4)
    np.testing.assert_almost_equal(res['monocrrlb'], 0.8041, 4)
    np.testing.assert_almost_equal(res['monocrrub'], 1.0054, 4)
    print_results(res)


def test_ace_balke_pearl_bounds_trivariate():
    # Mendelian randomization with 3 category instrument
    x = [0]*83 + [1]*5 + [0]*11 + [1]*1 + [0]*88 + [1]*6 + [0]*5 + [1]*1 + [0]*72 + [1]*5 + [0]*20 + [1]*3
    y = [0]*83 + [0]*5 + [1]*11 + [1]*1 + [0]*88 + [0]*6 + [1]*5 + [1]*1 + [0]*72 + [0]*5 + [1]*20 + [1]*3
    z = [0]*100 + [1]*100 + [2]*100
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    res = ace_balke_pearl_bounds(df['x'], df['y'], df['z'])
    assert res['nzcats'] == 3
    assert res['inequality']
    np.testing.assert_almost_equal(res['bplb'], -0.090, 4)
    np.testing.assert_almost_equal(res['bpub'], 0.74, 4)
    np.testing.assert_almost_equal(res['p11low'], 0.03, 4)
    np.testing.assert_almost_equal(res['p11upp'], 0.800, 4)
    np.testing.assert_almost_equal(res['p10low'], 0.06, 4)
    np.testing.assert_almost_equal(res['p10upp'], 0.12, 4)
    np.testing.assert_almost_equal(res['crrlb'], 0.25, 4)
    np.testing.assert_almost_equal(res['crrub'], 13.3333, 4)
    assert not res['monoinequality']
