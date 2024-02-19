import numpy as np
import matplotlib.pyplot as plt


def epsilon_greedy(theta, K, T):
    # Initialize
    s = [0]*K  # reward 받은 횟수
    f = [0]*K  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)

    epsilon = 0.1

    # Execute one run
    for t in range(T):
        # print(s, f)
        # explore할지 여부 선택
        exploring = int(np.random.rand() < epsilon)
            
        # Choose action
        if exploring:
            chosen_action = np.random.choice(range(K))
            action[t] = chosen_action # 0 ~ K-1
        else:
            zero_denominator_indices = np.where((np.array(s) + np.array(f)) == 0)[0]

            # If there are indices with a zero denominator, choose randomly among them
            if len(zero_denominator_indices) > 0:
                chosen_action = np.random.choice(zero_denominator_indices)
            else:

                chosen_action = np.argmax(np.array(s) / (np.array(s)+np.array(f)))
            
            action[t] = chosen_action   # 0 ~ K-1

        current_theta = theta[action[t],0]
        
        # Pull lever
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       
        
        # Record
        best_action = np.argmax([prob for prob in theta[:,0]])
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0
    
    return action, reward, prob

def UCB(theta, K, T):
    # Initialize
    s = [0]*K  # reward 받은 횟수
    f = [0]*K  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)

    # Execute one run
    for t in range(T):
        
        # Choose action
        # Find indices where the denominator is 0
        zero_denominator_indices = np.where((np.array(s) + np.array(f)) == 0)[0]

        # If there are indices with a zero denominator, choose randomly among them
        if len(zero_denominator_indices) > 0:
            chosen_action = np.random.choice(zero_denominator_indices)
        else:
            # Calculate UCB values
            ucb = (np.array(s) / (np.array(s) + np.array(f))) + 0.2*np.sqrt(2 * np.log(t + 1) / (np.array(s) + np.array(f)))
            chosen_action = np.argmax(ucb)
        
        action[t] = chosen_action

        # Pull lever
        current_theta = theta[action[t],0]
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       
        
        # Record
        best_action = np.argmax([prob for prob in theta[:,0]])
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0

    return action, reward, prob

def thompson(theta, K, T):
    # Initialize
    s = [1]*K  # reward 받은 횟수
    f = [1]*K  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)

    # Execute one run
    for t in range(T):
        
        # Choose action
        
        theta_hat = [np.random.beta(np.array(s), np.array(f))]
        # theta_hat = [np.random.beta(i, j) for i, j in zip(s, f)]
        chosen_action = np.argmax(theta_hat)
        action[t] = chosen_action

        # Pull lever
        current_theta = theta[action[t],0]
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       
        
        # Record
        best_action = np.argmax([prob for prob in theta[:,0]])
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0
    
    return action, reward, prob

if __name__=="__main__":

    # Define the parameters and data
    max_prob = 0.5
    K = 6
    epsilon = 0.3

    # theta = [max_prob - epsilon] * K
    # rand_idx = np.random.randint(0, K)
    # theta[rand_idx] = max_prob
    # theta = np.array(theta).reshape(-1, 1)

    # theta = np.array([[0.6], [0.5], [0.75], [0.45], [0.55]])
    theta = np.array([[0.85], [0.85], [0.65], [0.55], [0.50], [0.45]])

    T = 1000    # Number of time steps
    N = 1000    # Number of Monte Carlo Samples

    # Total number of algorithms
    # algorithms = ['epsilon_greedy']
    # algorithms = ['UCB']
    # algorithms = ['UCB', 'epsilon_greedy']
    # algorithms = ['UCB', 'thompson']
    algorithms = ['epsilon_greedy', 'UCB', 'thompson']
    n_algs = len(algorithms)

    # Arrays to store results
    action_sum = np.zeros((n_algs, T))
    reward_sum = np.zeros((n_algs, T))
    prob_sum = np.zeros((n_algs, T))

    # Plot settings
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    N_PLOTS = 2

    # Open figure
    plt.figure(figsize=(14, 5))

    for alg in range(n_algs):
        # Get handle of algorithm to run
        print(f'Running algorithm: {algorithms[alg]}')

        # Generate Monte Carlo simulations
        for n in range(N):
    
            # T step까지의 하나의 run을 얻어냄
            action, reward, prob = locals()[algorithms[alg]](theta, K, T)
            
            action_sum[alg, :] += action
            reward_sum[alg, :] += reward
            prob_sum[alg, :] += prob
            # print(prob)
            # Report progress
            if n % 100 == 0:
                print(f' Samples: {n}')

        # Monte Carlo estimates -> N개의 T step run들의 결과를 이용해서 p_action, regret을 평균으로 계산
        p_action = prob_sum[alg, :] / N   
        regret = np.max(theta[:, 0]) * np.arange(1, T+1) - np.cumsum(reward_sum[alg, :] / N)
        
        # Probability of Optimal Action Plot
        plt.subplot(1, N_PLOTS, 1)
        plt.plot(p_action, color=colors[alg], label=algorithms[alg])
        plt.title('Probability of Optimal Action')
        plt.xlabel('Trial')
        plt.ylabel('Probability')
        plt.ylim([0.0, 1.0])
        
        # Cum.regret Plot
        plt.subplot(1, N_PLOTS, 2)
        plt.plot(regret, color=colors[alg], label=algorithms[alg])
        plt.title('Regret')
        plt.xlabel('Trial')
        plt.ylabel('Cum. Regret')
        plt.ylim([-1, 100])

    plt.legend(loc='upper left')
    plt.show()