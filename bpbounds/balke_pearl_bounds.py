import numpy as np
import pandas as pd

from typing import Dict
from bpbounds.bpbounds_calc import bpbounds_tri_x2y2z2, bpbounds_tri_x2y2z3, bpbounds_calc_tri_z3, bpbounds_calc_tri_z2


# Nonparametric Bounds for the Average Causal Effect due to Balke and Pearl
# (https://www.tandfonline.com/doi/abs/10.1080/01621459.1997.10474074).

def ace_balke_pearl_bounds(x: pd.Series, y: pd.Series, z: pd.Series) -> Dict:
    x_len = len(x.unique())
    y_len = len(y.unique())
    z_len = len(z.unique())
    if x_len != 2 or y_len != 2:
        print("x and y must be binary!")

    if z_len > 3:
        print("z must be binary or ternary!")

    # convert to distributions
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    p = np.zeros((z_len, y_len, x_len))
    for _z in df['z'].unique():
        stratified_df = df[df['z'] == _z]
        total_sum = stratified_df[['x', 'y']].shape[0]
        xy_values = stratified_df[['x', 'y']].value_counts().reset_index()
        for i_x in stratified_df['x'].unique():
            if not np.isnan(i_x):
                for i_y in stratified_df['y'].unique():
                    p[int(_z)][int(i_y)][int(i_x)] = \
                        xy_values[(xy_values['x'] == i_x) & (xy_values['y'] == i_y)][0].reset_index()[0] / total_sum

    # check if p conditional probabilities
    for _z in df['z'].unique():
        if not np.isnan(_z) and p[int(_z)].sum() < 1:
            print('p does not exactly add up to 1')

    if len(p) == 2:
        bp = bpbounds_tri_x2y2z2(p=p)
        bpres = bpbounds_calc_tri_z2(p)
    elif len(p) == 3:
        bp = bpbounds_tri_x2y2z3(p=p)
        bpres = bpbounds_calc_tri_z3(p)
    else:
        return {}

    retlist = {
        "nzcats": len(z.unique()),
        "inequality": bp['inequality'],
        "bplb": bp['bplow'],
        "bpub": bp['bpupp'],
        "bplower": bp['bplower'],
        "bpupper": bp['bpupper'],
        "p11low": bpres['p11low'],
        "p11upp": bpres['p11upp'],
        "p10low": bpres['p10low'],
        "p10upp": bpres['p10upp'],
        "p11lower": bpres['p11lower'],
        "p11upper": bpres['p11upper'],
        "p10lower": bpres['p10lower'],
        "p10upper": bpres['p10upper'],
        "crrlb": bpres['crrlb'],
        "crrub": bpres['crrub'],
        "monoinequality": bpres['monoinequality'],
        "monobplb": bpres['monobplb'],
        "monobpub": bpres['monobpub'],
        "monop11low": bpres['monop11lb'],
        "monop11upp": bpres['monop11ub'],
        "monop10low": bpres['monop10lb'],
        "monop10upp": bpres['monop10ub'],
        "monocrrlb": bpres['monocrrlb'],
        "monocrrub": bpres['monocrrub']
    }
    return retlist


def print_results(result_dict: Dict):
    print(f'Instrument categories:   {result_dict["nzcats"]}')
    print(f'Instrumental inequality: {result_dict["inequality"]}')
    if result_dict['inequality']:
        print(f'Causal parameter Lower bound Upper bound')
        print(f'            ACE  {result_dict["bplb"]:.7f} {result_dict["bpub"]:.7f}')
        print(f'   P(Y|do(X=0))  {result_dict["p10low"]:.7f} {result_dict["p10upp"]:.7f}')
        print(f'   P(Y|do(X=1))  {result_dict["p11low"]:.7f} {result_dict["p11upp"]:.7f}')
        print(f'            CRR  {result_dict["crrlb"]:.7f} {result_dict["crrub"]:.7f}')

    print(f'Monotonicity inequality: {result_dict["monoinequality"]}')
    if result_dict['monoinequality']:
        print(f'Causal parameter Lower bound Upper bound')
        print(f'            ACE  {result_dict["monobplb"]:.7f} {result_dict["monobpub"]:.7f}')
        print(f'   P(Y|do(X=0))  {result_dict["monop10low"]:.7f} {result_dict["monop10upp"]:.7f}')
        print(f'   P(Y|do(X=1))  {result_dict["monop11low"]:.7f} {result_dict["monop11upp"]:.7f}')
        print(f'            CRR  {result_dict["monocrrlb"]:.7f} {result_dict["monocrrub"]:.7f}')
