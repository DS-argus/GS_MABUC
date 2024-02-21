from itertools import product
from collections import Counter
from npsem.model import CD, SCM
from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt


class CausalBandit:

    def __init__(self, G: CD, M: SCM, ISs: Iterable[set], reward_v: str) -> None:
        self.G = G
        self.M = M
        self.order = G.causal_order()
        self.ISs = ISs
        self.reward_v = reward_v
        self.expected_rewards, self.optimal_arm, self.optimal_reward = self.calc_Er()
        self.all_actions = list(self.expected_rewards.keys())

    @staticmethod
    def compare_regrets(regrets: dict[str: np.array]):
        """
        여러 알고리즘의 cumulative regret을 받아서 같이 plot해주는 함수
        """
        for name, regret in regrets.items():
            plt.plot(regret, label=name)
        
        plt.legend()
        plt.show()


    def calc_Er(self):
        """
        SCM을 이용해 실제 optimal arm과 expected reward 계산 -> regret 계산에 필요
        """
        expected_rewards = defaultdict(int)
        for IS in self.ISs:
            for comb in product(*[self.M.D[V] for V in IS]):
                intervention = dict(zip(IS, comb))
                Er = self.M.query(outcome=(self.reward_v,), intervention=intervention)[(1,)]

                action = tuple(intervention.items())
                expected_rewards[action] = Er

        optimal_arm = max(expected_rewards, key=expected_rewards.get)
        optimal_reward = expected_rewards[optimal_arm]

        return expected_rewards, optimal_arm, optimal_reward


    def UCB1(self, T: int, average: int, verbose: bool = False):

        print("[Running UCB1 algorithm]")
        
        if verbose: 
            print(f"\texpected rewards: \t{self.expected_rewards}")
            print(f"\ttheoretical optimal arm: {self.optimal_arm}\n\ttheoretical optimal reward: {self.optimal_reward}")
        
        # action 기록, Vs(Y) 기록
        rewards = np.zeros((average, T))
        action_results = np.zeros((average, T)) # self.all_actions의 index를 저장
        estimation_results = np.zeros((average, 2))
        for i in range(average):
            
            # Running algorithm (intervention index) : [N, 1(Y=1), Q_hat, UCB_hat]
            estimated_reward = {k: [1, 0, 0, float('inf')] for k in range(len(self.all_actions))}
            Vs_results = np.zeros((T, len(self.order)))   # 추후 observation으로 ID를 estimate할 때 필요함
            for t in range(T):
                
                # UCB로 arm 선택하고 기록
                action_t = max(estimated_reward, key=lambda k: estimated_reward[k][-1])
                action_results[i][t] = action_t

                # action_t에 따른 변수 Vs, reward 관찰하고 기록
                obs_Vs = self.M.sample(1, self.all_actions[action_t], user_order=self.order)
                Vs_results[t,:] = obs_Vs[0]
                
                reward = obs_Vs[0][self.order.index(self.reward_v)]
                rewards[i][t] = reward

                # 각 arm의 정보 업데이트
                for k in estimated_reward.keys():
                    if k == action_t:
                        estimated_reward[k][0] += 1
                        estimated_reward[k][1] += reward
                        estimated_reward[k][2] = estimated_reward[k][1] / estimated_reward[k][0]

                    estimated_reward[k][3] = estimated_reward[k][2] + np.sqrt(2*np.log(t+1)/estimated_reward[k][0])

            # 하나의 반복 종료 : estimated arms와 그 estimated rewards 저장
            estimated_optimal_arm = max(estimated_reward, key=lambda k: estimated_reward[k][2])
            estimated_optimal_reward = estimated_reward[estimated_optimal_arm][2]
            estimation_results[i,:] = [estimated_optimal_arm, estimated_optimal_reward]

        # 모든 반복 종료 -> cumulative regret 계산
        final = np.mean(rewards, axis=0)
        cumulative_regret = np.array([t*self.optimal_reward for t in range(T)]) - np.cumsum(final)
        
        # 실제 optimal arm, reward와 추정된 optimal arm, reward 출력
        estimated_optimal_arm, _ = Counter(estimation_results[:, 0]).most_common(1)[0]
        filtered_rows = estimation_results[estimation_results[:, 0] == estimated_optimal_arm]
        estimated_optimal_reward = np.mean(filtered_rows[:, 1])

        if verbose:
            print(f"\testimated optimal arm: {self.all_actions[int(estimated_optimal_arm)]}\n\testimated optimal reward: {estimated_optimal_reward}")

        return cumulative_regret


    def TS(self, ISs: Iterable[set], reward_v: str, T: int, average: int, plot: bool = True):
        
        # action에 대해 E(r) 계산해서 optimal arm 찾기 : regret계산 위해 필요함
        expected_rewards = defaultdict(int)

        values = self.M.all_values()
        for IS in ISs:
            for comb in product(*[values[V] for V in IS]):
                intervention = dict(zip(IS, comb))

                Er = self.M.query(outcome=(reward_v), intervention=intervention)[(1,)]
                action = (tuple(intervention.keys()), tuple(intervention.values()))
                expected_rewards[action] = Er

        optimal_arm = max(expected_rewards, key=expected_rewards.get)
        optimal_reward = expected_rewards[optimal_arm]
        print(f"expected_rewards: {expected_rewards}")

        print("[Running TS algorithm]")
        # Running algorithm ('action', value) : [선택한 횟수, reward 받은 횟수, TS_estimate]
        results = np.zeros((average, T))
        for i in range(average):

            estimated_reward = {k: [2, 1, np.random.beta(1, 1)] for k in expected_rewards.keys()}
            for T in range(T):
                
                # TS로 arm 선택
                action_t = max(estimated_reward, key=lambda k: estimated_reward[k][-1])

                # action_t에 따른 결과 
                observation = self.observe_action(action_t)

                # 결과 저장
                results[i][T] = observation[reward_v]

                # action arm의 정보 업데이트
                estimated_reward[action_t][0] += 1
                estimated_reward[action_t][1] += observation[reward_v]
                for k in estimated_reward.keys():
                    estimated_reward[k][2] = np.random.beta(estimated_reward[k][1], estimated_reward[k][0]-estimated_reward[k][1])      # beta(reward 받은 횟수, reward 못받은 횟수)

        final = np.mean(results, axis=0)
        cumulative_regret = np.array([t*optimal_reward for t in range(T)]) - np.cumsum(final)

        print(f"theoretical optimal arm: {optimal_arm}\n theoretical optimal reward: {optimal_reward}")

        if plot:
            plt.plot(cumulative_regret)
            plt.show()

        return cumulative_regret
    

if __name__ == "__main__":
    G = CD({'X1', 'X2', 'Y'}, [('X1', 'Y'), ('X2', 'Y')])

    def P_U(vs):
        p_u = 1.
        p_u *= 0.9 if vs['U_X1']==1 else 0.1
        p_u *= 0.1 if vs['U_X2']==1 else 0.9
        p_u *= 0.4 if vs['U_Y']==1 else 0.6
        return p_u
    
    domains = {'X1': (0, 1),
               'X2': (0, 1),
               'Y': (0, 1),
               'U_X1': (0, 1),
               'U_X2': (0, 1),
               'U_Y': (0, 1)
               }
    
    M = SCM(G, 
            F={
                'X1': lambda vs: vs['U_X1'],
                'X2': lambda vs: vs['U_X2'],
                'Y': lambda vs: vs['U_Y'] ^ vs['X1'] & vs['X2']
            },
            P_U = P_U,
            D = domains,
            more_U={'U_X1', 'U_X2', 'U_Y'}
            )


    # actions = [{'X1'}, {'X2'}, {'X1', 'X2'}]
    actions = [{'X1'}, {'X2'}]
    # actions = [{'X1', 'X2'}]

    Bandit = CausalBandit(G, M, ISs=actions, reward_v='Y')

    UCB_regret = Bandit.UCB1(T=1000, average=1000, verbose=True)
    # TS_regret = Bandit.TS(actions, reward_v='Y', T=1000, average=1000, plot = False)
    
    regrets = {'UCB_regret': UCB_regret,
               }
    
    Bandit.compare_regrets(regrets)

    # regrets = {'UCB_regret': UCB_regret,
    #            'TS_regret': TS_regret
    #            }
    
    # Bandit.compare_regrets(regrets)


