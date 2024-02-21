from model import CausalDiagram, default_P_U, defaultdict, StructuralCausalModel, cd2qcd
from utils import rand_bw, seeded
import networkx as nx
from where_do import CC, MISs, subMISs, bruteforce_POMISs, MUCT, IB, MUCT_IB, POMISs, subPOMISs, minimal_do
from scm_bandits import SCM_to_bandit_machine
from itertools import permutations, combinations

G1 = CausalDiagram({'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'}, 
                    [('X1', 'X2'), ('X1', 'X3'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X4'), ('X3', 'X5'), ('X4', 'X5'), ('X5', 'X6'), ('X4', 'X6'), ('X6', 'Y')], 
                    [])

# G2 = CausalDiagram({'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Y'}, 
#                     [('X1', 'X2'), ('X1', 'X3'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X4'), ('X3', 'X5'), ('X3', 'X6'), ('X6', 'X7'), ('X5', 'X7'), ('X7', 'Y')], 
#                     [])


G2 = CausalDiagram({'X1', 'Z1', 'Y1'}, 
                    [('X1', 'Z1'), ('Z1', 'Y1')], 
                    [('X1', 'Z1', 'U_X1Z1')])

# print(G2.UCs('X1')) 
# domains = defaultdict(lambda: (0, 1))   # binary q변수 추가하는 거
# domains['x']



def XYZWST(u_wx='U0', u_yz='U1'):
    W, X, Y, Z, S, T = 'W', 'X', 'Y', 'Z', 'S', 'T'
    return CausalDiagram({'W', 'X', 'Y', 'Z', 'S', 'T'}, [(Z, X), (X, Y), (W, Y), (S, W), (T, X), (T, Y)], [(X, W, u_wx), (Z, Y, u_yz)])


def XYZWST_SCM(devised=True, seed=None):
    with seeded(seed):
        G = XYZWST('U_WX', 'U_YZ')

        # parametrization for U
        if devised:
            mu1 = {'U_WX': rand_bw(0.4, 0.6, precision=2),
                   'U_YZ': rand_bw(0.4, 0.6, precision=2),
                   'U_X': rand_bw(0.01, 0.1, precision=2),
                   'U_Y': rand_bw(0.01, 0.1, precision=2),
                   'U_Z': rand_bw(0.01, 0.1, precision=2),
                   'U_W': rand_bw(0.01, 0.1, precision=2),
                   'U_S': rand_bw(0.1, 0.9, precision=2),
                   'U_T': rand_bw(0.1, 0.9, precision=2)
                   }
        else:
            mu1 = {'U_WX': rand_bw(0.01, 0.99, precision=2),
                   'U_YZ': rand_bw(0.01, 0.99, precision=2),
                   'U_X': rand_bw(0.01, 0.99, precision=2),
                   'U_Y': rand_bw(0.01, 0.99, precision=2),
                   'U_Z': rand_bw(0.01, 0.99, precision=2),
                   'U_W': rand_bw(0.01, 0.99, precision=2),
                   'U_S': rand_bw(0.01, 0.99, precision=2),
                   'U_T': rand_bw(0.01, 0.99, precision=2),
                   }

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(G,
                                  F={
                                      'S': lambda v: v['U_S'],
                                      'T': lambda v: v['U_T'],
                                      'W': lambda v: v['U_W'] ^ v['U_WX'] ^ v['S'],
                                      'Z': lambda v: v['U_Z'] ^ v['U_YZ'],
                                      'X': lambda v: 1 ^ v['U_X'] ^ v['Z'] ^ v['U_WX'] ^ v['T'],
                                      'Y': lambda v: v['U_Y'] ^ v['U_YZ'] ^ v['X'] ^ v['W'] ^ v['T']
                                  },
                                  P_U=default_P_U(mu1),
                                  D=domains,
                                  more_U={'U_W', 'U_X', 'U_Y', 'U_Z', 'U_S', 'U_T'})
        return M, mu1

if __name__ == "__main__":

    model, mu = XYZWST_SCM(True, 0)

    print(mu)
    print(model.query(outcome=('Y'), condition={'W': 1}, intervention={'X': 1}, verbose=True))