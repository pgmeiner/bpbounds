from typing import Dict
import numpy as np

from bpbounds.bp_params import A_tri_x2y2z2, ice_tri_x2y2z2, A_tri_x2y2z3, ice_tri_x2y2z3, cons_tri_x2y2z3


def bpbounds_tri_x2y2z2(p: np.array, A: np.array = A_tri_x2y2z2, ice: np.array = ice_tri_x2y2z2) -> Dict:
    prod = A @ p.ravel()
    ivinequality = prod[ice == 0]
    low = -1 * prod[ice == 1]
    upp = prod[ice == -1]
    inequality = min(ivinequality) >= 0
    bplow = max(low)
    bpupp = min(upp)

    return {"inequality": inequality,
            "bplow": bplow,
            "bpupp": bpupp,
            "bplower": low,
            "bpupper": upp}


def bpbounds_tri_x2y2z3(p: np.array, A: np.array = A_tri_x2y2z3, ice: np.array = ice_tri_x2y2z3,
                        cons: np.array = cons_tri_x2y2z3) -> Dict:
    prod = A @ p.ravel()
    prod = prod + cons
    ivinequality = prod[ice == 0]
    low = prod[ice == -1]
    upp = -1 * prod[ice == 1]
    inequality = max(ivinequality) <= 0
    bplow = max(low)
    bpupp = min(upp)

    return {"inequality": inequality,
            "bplow": bplow,
            "bpupp": bpupp,
            "bplower": low,
            "bpupper": upp}


def bpbounds_calc_tri_z3(p: np.array):
    # assuming that p is in the following order
    # notation for conditional probabilities is p(y,x|z)
    p = p.ravel()
    p000 = p[0]
    p100 = p[1]
    p010 = p[2]
    p110 = p[3]
    p001 = p[4]
    p101 = p[5]
    p011 = p[6]
    p111 = p[7]
    p002 = p[8]
    p102 = p[9]
    p012 = p[10]
    p112 = p[11]

    # bounds on probabilities
    p10low1 = p100
    p10low2 = p101
    p10low3 = p102
    p10low4 = p100 + p110 + p101 + p011 - 1
    p10low5 = p100 + p010 + p101 + p111 - 1
    p10low6 = p101 + p111 + p102 + p012 - 1
    p10low7 = p101 + p011 + p102 + p112 - 1
    p10low8 = p102 + p112 + p100 + p010 - 1
    p10low9 = p102 + p012 + p100 + p110 - 1
    p10upp1 = 1 - p000
    p10upp2 = 1 - p001
    p10upp3 = 1 - p002
    p10upp4 = p100 + p010 + p101 + p111
    p10upp5 = p100 + p110 + p101 + p011
    p10upp6 = p101 + p011 + p102 + p112
    p10upp7 = p101 + p111 + p102 + p012
    p10upp8 = p102 + p012 + p100 + p110
    p10upp9 = p102 + p112 + p100 + p010
    p10low = max([p10low1, p10low2, p10low3, p10low4, p10low5, p10low6, p10low7, p10low8, p10low9])
    p10upp = min([p10upp1, p10upp2, p10upp3, p10upp4, p10upp5, p10upp6, p10upp7, p10upp8, p10upp9])
    p11low1 = p110
    p11low2 = p111
    p11low3 = p112
    p11low4 = p100 + p110 - p101 - p011
    p11low5 = -p100 - p010 + p101 + p111
    p11low6 = p101 + p111 - p102 - p012
    p11low7 = -p101 - p011 + p102 + p112
    p11low8 = p102 + p112 - p100 - p010
    p11low9 = -p102 - p012 + p100 + p110
    p11upp1 = 1 - p010
    p11upp2 = 1 - p011
    p11upp3 = 1 - p012
    p11upp4 = p100 + p110 - p101 - p011 + 1
    p11upp5 = -p100 - p010 + p101 + p111 + 1
    p11upp6 = p101 + p111 - p102 - p012 + 1
    p11upp7 = -p101 - p011 + p102 + p112 + 1
    p11upp8 = p102 + p112 - p100 - p010 + 1
    p11upp9 = -p102 - p012 + p100 + p110 + 1

    p11low = max([p11low1, p11low2, p11low3, p11low4, p11low5, p11low6, p11low7, p11low8, p11low9])
    p11upp = min([p11upp1, p11upp2, p11upp3, p11upp4, p11upp5, p11upp6, p11upp7, p11upp8, p11upp9])
    p10lower = [p10low1, p10low2, p10low3, p10low4, p10low5, p10low6, p10low7, p10low8, p10low9]
    p10upper = [p10upp1, p10upp2, p10upp3, p10upp4, p10upp5, p10upp6, p10upp7, p10upp8, p10upp9]
    p11lower = [p11low1, p11low2, p11low3, p11low4, p11low5, p11low6, p11low7, p11low8, p11low9]
    p11upper = [p11upp1, p11upp2, p11upp3, p11upp4, p11upp5, p11upp6, p11upp7, p11upp8, p11upp9]
    retlist = {"p10low": p10low,
               "p10upp": p10upp,
               "p11low": p11low,
               "p11upp": p11upp,
               "p10lower": p10lower,
               "p10upper": p10upper,
               "p11lower": p11lower,
               "p11upper": p11upper}

    # bounds on causal risk ratio
    rrlow = p11low / p10upp
    rrupp = p11upp / p10low
    retlist["crrlb"] = rrlow
    retlist["crrub"] = rrupp

    # monotonicity bounds
    m1 = (p102 <= p101) & (p101 <= p100)
    m2 = (p110 <= p111) & (p111 <= p112)
    m3 = (p010 <= p011) & (p011 <= p012)
    m4 = (p002 <= p001) & (p001 <= p000)
    monoinequality = (m1 == True & m2 == True & m3 == True & m4 == True)
    retlist["monoinequality"] = monoinequality
    if monoinequality:
        mlow = p112 + p000 - 1
        mupp = 1 - p100 - p110
        retlist["monobplb"] = mlow
        retlist["monobpub"] = mupp

        # bounds on intervention probabilities assuming monotonicity
        monop10low = p100
        monop10upp = 1 - p000
        monop11low = p112
        monop11upp = 1 - p012
        retlist["monop10lb"] = monop10low
        retlist["monop10ub"] = monop10upp
        retlist["monop11lb"] = monop11low
        retlist["monop11ub"] = monop11upp

        # bounds on causal risk ratio assuming monotonicity
        monocrrlow = monop11low / monop10upp
        monocrrupp = monop11upp / monop10low
        retlist["monocrrlb"] = monocrrlow
        retlist["monocrrub"] = monocrrupp
    else:
        retlist["monobplb"] = 0
        retlist["monobpub"] = 0
        retlist["monop10lb"] = 0
        retlist["monop10ub"] = 0
        retlist["monop11lb"] = 0
        retlist["monop11ub"] = 0
        retlist["monocrrlb"] = 0
        retlist["monocrrub"] = 0
    return retlist


def bpbounds_calc_tri_z2(p: np.array) -> Dict:
    # assuming that p is in the following order
    # notation for conditional probabilities is p(y,x|z)
    p = p.ravel()
    p000 = p[0]
    p100 = p[2]
    p010 = p[1]
    p110 = p[3]
    p001 = p[4]
    p101 = p[6]
    p011 = p[5]
    p111 = p[7]

    # pearl bounds on probabilities
    p10low1 = p101
    p10low2 = p100
    p10low3 = p100 + p110 - p001 - p111
    p10low4 = p010 + p100 - p001 - p011
    p10upp1 = 1 - p001
    p10upp2 = 1 - p000
    p10upp3 = p010 + p100 + p101 + p111
    p10upp4 = p100 + p110 + p011 + p101
    p10low = max([p10low1, p10low2, p10low3, p10low4])
    p10upp = min([p10upp1, p10upp2, p10upp3, p10upp4])
    p11low1 = p110
    p11low2 = p111
    p11low3 = -p000 - p010 + p001 + p111
    p11low4 = -p010 - p100 + p101 + p111
    p11upp1 = 1 - p011
    p11upp2 = 1 - p010
    p11upp3 = p000 + p110 + p101 + p111
    p11upp4 = p100 + p110 + p001 + p111
    p11low = max([p11low1, p11low2, p11low3, p11low4])
    p11upp = min([p11upp1, p11upp2, p11upp3, p11upp4])

    p10lower = [p10low1, p10low2, p10low3, p10low4]
    p10upper = [p10upp1, p10upp2, p10upp3, p10upp4]
    p11lower = [p11low1, p11low2, p11low3, p11low4]
    p11upper = [p11upp1, p11upp2, p11upp3, p11upp4]

    retlist = {
        "p10low": p10low,
        "p10upp": p10upp,
        "p11low": p11low,
        "p11upp": p11upp,
        "p10lower": p10lower,
        "p10upper": p10upper,
        "p11lower": p11lower,
        "p11upper": p11upper
    }

    # bounds on causal risk ratio
    rrlow = p11low / p10upp
    rrupp = p11upp / p10low
    retlist["crrlb"] = rrlow
    retlist["crrub"] = rrupp

    # monotonicity bounds
    m1 = p000 - p001 >= 0
    m2 = p011 - p010 >= 0
    m3 = p100 - p101 >= 0
    m4 = p111 - p110 >= 0
    mlow = p000 - p001 - p011 - p101
    mupp = p000 + p010 + p110 - p011
    monoinequality = (m1 == True & m2 == True & m3 == True & m4 == True)
    if monoinequality:
        retlist["monobplb"] = mlow
        retlist["monobpub"] = mupp

        # bounds on intervention probabilities assuming monotonicity
        monop10low = p100
        monop10upp = 1 - p000
        monop11low = p111
        monop11upp = 1 - p011
        retlist["monop10lb"] = monop10low
        retlist["monop10ub"] = monop10upp
        retlist["monop11lb"] = monop11low
        retlist["monop11ub"] = monop11upp

        # bounds on causal risk ratio assuming monotonicity
        monocrrlow = monop11low / monop10upp
        monocrrupp = monop11upp / monop10low
        retlist["monocrrlb"] = monocrrlow
        retlist["monocrrub"] = monocrrupp
    else:
        retlist["monobplb"] = 0
        retlist["monobpub"] = 0
        retlist["monop10lb"] = 0
        retlist["monop10ub"] = 0
        retlist["monop11lb"] = 0
        retlist["monop11ub"] = 0
        retlist["monocrrlb"] = 0
        retlist["monocrrub"] = 0

    retlist["monoinequality"] = monoinequality
    return retlist
