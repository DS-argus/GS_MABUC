from itertools import product
from utils import rand_bw, seeded
from collections import defaultdict
from model import CausalDiagram, StructuralCausalModel, default_P_U, dict_except
from typing import Dict, Iterable, Optional, Set, Sequence, AbstractSet, FrozenSet, Tuple

import numpy as np
import random as rnd
import matplotlib.pyplot as plt


class CausalBandit:

    def __init__(self, G: "CausalDiagram", M: "StructuralCausalModel", mu1: dict) -> None:
        self.G = G
        self.M = M
        self.mu1 = mu1                      # data generation할 때 U의 확률분포가 필요


    def observe_action(self, action):
        # generate U
        u = [int(rnd.random() < p) for u, p in mu1.items()]
        assigned = dict(zip(mu1.keys(), u))

        V_ordered = self.G.causal_order()

        for V_i in V_ordered:
            if V_i in action[0]:
                assigned[V_i] = action[1][action[0].index(V_i)]
            else:
                assigned[V_i] = self.M.F[V_i](assigned)
        
        return assigned
        
        
    def UCB1(self, ISs: Iterable[Tuple], reward_v: str, round: int, average: int):

        # action에 대해 E(r) 계산 
        expected_reward = defaultdict(int)

        values = self.M.all_values()
        for IS in ISs:
            for comb in product(*[values[V] for V in IS]):
                intervention = dict(zip(IS, comb))

                Er = self.M.query(outcome=(reward_v), intervention=intervention)[(1,)]
                action = (tuple(intervention.keys()), tuple(intervention.values()))
                expected_reward[action] = Er


        optimal_arm = max(expected_reward, key=expected_reward.get)
        optimal_reward = expected_reward[optimal_arm]

        # Running algorithm ('action', value) : [N, 1(Y=1), p_hat, UCB_hat]
        results = np.zeros((average, round))
        print(expected_reward)
        for i in range(average):

            estimated_reward = {k: [0, 0, 0, float('inf')] for k in expected_reward.keys()}
            for T in range(round):
                
                # UCB로 arm 선택
                action_t = max(estimated_reward, key=lambda k: estimated_reward[k][-1])

                # action_t에 따른 결과 
                observation = self.observe_action(action_t)

                # 결과 저장
                results[i][T] = observation[reward_v]

                # 각 arm의 정보 업데이트
                for k, v in estimated_reward.items():
                    if k == action_t:
                        estimated_reward[k][0] += 1
                        estimated_reward[k][1] += observation[reward_v]
                        estimated_reward[k][2] = estimated_reward[k][1] / estimated_reward[k][0]

                    if estimated_reward[k][0] != 0:
                        estimated_reward[k][3] = estimated_reward[k][2] + np.sqrt(2*np.log(T+1)/estimated_reward[k][0])


        final = np.mean(results, axis=0)
        cumulative_regret = np.array([t*optimal_reward for t in range(round)]) - np.cumsum(final)

        print(f"optimal arm: {optimal_arm}\n optimal reward: {optimal_reward}")
        plt.plot(cumulative_regret)
        plt.show()
    
if __name__ == "__main__":
    # G = CausalDiagram({'X1', 'X2', 'Y'}, [('X1', 'Y'), ('X2', 'Y')])

    # mu1 = {'U_X1' : 0.6,
    #        'U_X2' : 0.8,
    #        'U_Y' : 0.4
    #        }
    
    # domains = {'U_X1': (0, 1),
    #            'U_X2': (0, 1),
    #            'U_Y': (0, 1)
    #            }
    
    # M = StructuralCausalModel(G, 
    #                           F={
    #                               'X1': lambda v: v['U_X1'],
    #                               'X2': lambda v: v['U_X2'],
    #                               'Y': lambda v: v['U_Y'] ^ v['X1'] ^ v['X2']
    #                           },
    #                           P_U = default_P_U(mu1),
    #                           D = domains,
    #                           more_U={'U_X1', 'U_X2', 'U_Y'}
    #                           )

    G = CausalDiagram({'X1', 'Y'}, [('X1', 'Y')])

    mu1 = {'U_X1' : 0.8,
           'U_Y' : 0.2
           }
    
    domains = {'U_X1': (0, 1),
               'U_Y': (0, 1)
               }
    
    M = StructuralCausalModel(G, 
                              F={
                                  'X1': lambda v: v['U_X1'],
                                  'Y': lambda v: v['U_Y'] ^ v['X1']
                              },
                              P_U = default_P_U(mu1),
                              D = domains,
                              more_U={'U_X1', 'U_Y'}
                              )
    
    Bandit = CausalBandit(G, M, mu1)

    actions = [['X1']]

    Bandit.UCB1(actions, reward_v='Y', round=1000, average=1000)
