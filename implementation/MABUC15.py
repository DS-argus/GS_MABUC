import sys
from pathlib import Path

# Add the parent directory to sys.path to be able to import model
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from model import CausalDiagram,  StructuralCausalModel, default_P_U
from causalbandit import CausalBandit
import numpy as np
import random as rnd

# 논문에서 제시된 parameterization을 갖는 SCM을 찾을 수 없어서 직접 parametrizaion해줘야 함

if __name__ == "__main__":

    G = CausalDiagram({'X', 'Y'}, [('X', 'Y')], [('X', 'Y', 'U_XY')])

    mu1 = {'U_X' : 0.5,
           'U_Y' : 0.45,
           'U_XY': 0.6
           }
    
    domains = {'U_X': (0, 1),
               'U_Y': (0, 1),
               'U_XY': (0, 1)
               }
    
    M = StructuralCausalModel(G, 
                              F={
                                  'X': lambda v: v['U_X'] ^ v['U_XY'],
                                  'Y': lambda v: v['U_Y'] ^ v['X'] ^ v['U_XY']
                              },
                              P_U = default_P_U(mu1),
                              D = domains,
                              more_U={'U_X', 'U_Y'}
                              )

    actions = [['X']]

    Bandit = CausalBandit(G, M, mu1)
    UCB_regret = Bandit.UCB1(actions, reward_v='Y', round=10000, average=1000, plot = False)
    TS_regret = Bandit.TS(actions, reward_v='Y', round=10000, average=1000, plot = False)

    regrets = {'UCB_regret': UCB_regret,
               'TS_regret': TS_regret
               }
    
    Bandit.compare_regrets(regrets)

