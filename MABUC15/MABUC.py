import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(theta, T, all_factors, pObs):
    # Initialize
    s = [0, 0]  # 누적보상합 = reward 받은 횟수
    f = [0, 0]  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    epsilon = 0.3

    # Execute one run
    for t in range(T):
        
        # get U
        round_factors = all_factors[:, t]
        B = round_factors[0]
        D = round_factors[1]
        
        # combinations of covariates : # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
        covariate_index = int(B + D * 2)    
        conds[covariate_index] += 1
        
        # explore할지 여부 선택
        exploring = int(np.random.rand() < epsilon)
        
        # Choose action
        if exploring:
            chosen_action = int(np.random.rand() < 0.5)
            action[t] = chosen_action # 0 or 1
        else:
            denom_0 = s[0] + f[0] + 1e-6
            denom_1 = s[1] + f[1] + 1e-6
            chosen_action = np.argmax([s[0] / denom_0, s[1] / denom_1])
            action[t] = chosen_action   # 0 or 1
        
        # 현재 B, D, X하에서 payout rates
        current_theta = theta[action[t], covariate_index]
        
        # Pull lever
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       # reward 못 받았으면 ++
        
        # Record
        best_action = np.argmax([theta[0, covariate_index], theta[1, covariate_index]]) # 0 or 1
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0
    
    return action, reward, prob, conds

def UCB(theta, T, all_factors, pObs):
    
    # Initialize
    s = [0, 0]  # 누적보상합 = reward 받은 횟수
    f = [0, 0]  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    # Execute one run
    for t in range(T):
        
        # get U
        round_factors = all_factors[:, t]
        B = round_factors[0]
        D = round_factors[1]
        
        # combinations of covariates : # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
        covariate_index = int(B + D * 2)    
        conds[covariate_index] += 1
        
        # choose action
        ucb = np.array(s) / (np.array(s)+np.array(f)) + 0.2*np.sqrt(2*np.log(t+1)/(np.array(s)+np.array(f)))
        chosen_action = np.argmax(ucb)
        action[t] = chosen_action
        
        # Pull lever
        current_theta = theta[action[t], covariate_index]
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       # reward 못받았으면 ++
        
        # Record
        best_action = np.argmax([theta[0, covariate_index], theta[1, covariate_index]]) # 0 or 1
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0
    
    return action, reward, prob, conds

def observational(theta, T, all_factors, pObs):
    # Initialize
    s = [0, 0]  # 누적보상합 = reward 받은 횟수
    f = [0, 0]  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    # Execute one run
    for t in range(T):
        
        # get U
        round_factors = all_factors[:, t]
        B = round_factors[0]
        D = round_factors[1]
        Z = round_factors[2]

        # combinations of covariates : # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
        covariate_index = int(B + D * 2)    
        conds[covariate_index] += 1
        
        # choose action  Z = XOR(D, B)
        action_choice = Z
        action[t] = int(action_choice)
        
        # Pull lever
        current_theta = theta[action[t], covariate_index]
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       # reward 못받았으면 ++
        
        # Record
        best_action = np.argmax([theta[0, covariate_index], theta[1, covariate_index]]) # 0 or 1
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0

    return action, reward, prob, conds

def coin_flipping(theta, T, all_factors, pObs):
    # Initialize
    s = [0, 0]  # 누적보상합 = reward 받은 횟수
    f = [0, 0]  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    # Execute one run
    for t in range(T):
        
        # get U
        round_factors = all_factors[:, t]
        B = round_factors[0]
        D = round_factors[1]

        # combinations of covariates : # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
        covariate_index = int(B + D * 2)    
        conds[covariate_index] += 1
        
        # choose action  Z = XOR(D, B)
        action_choice = int(np.random.rand() < 0.5)
        action[t] = int(action_choice)
        
        # Pull lever
        current_theta = theta[action[t], covariate_index]
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       # reward 못받았으면 ++
        
        # Record
        best_action = np.argmax([theta[0, covariate_index], theta[1, covariate_index]]) # 0 or 1
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0

    return action, reward, prob, conds

def thompson(theta, T, all_factors, pObs):
    
    # Initialize
    s = [1, 1]  # 누적보상합 = reward 받은 횟수
    f = [1, 1]  # reward 못받은 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    # Execute one run
    for t in range(T):
        
        # get U
        round_factors = all_factors[:, t]
        B = round_factors[0]
        D = round_factors[1]
        
        # combinations of covariates : # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
        covariate_index = int(B + D * 2)    
        conds[covariate_index] += 1
        
        # choose action
        theta_hat = [np.random.beta(s[0], f[0]), np.random.beta(s[1], f[1])]
        chosen_action = np.argmax(theta_hat)
        action[t] = chosen_action
        
        # Pull lever
        current_theta = theta[action[t], covariate_index]
        reward_choice = int(np.random.rand() <= current_theta)      # 0 or 1
        reward[t] = reward_choice
        
        # Update
        s[action[t]] += reward_choice
        f[action[t]] += 1 - reward_choice       # reward 못받았으면 ++
        
        # Record
        best_action = np.argmax([theta[0, covariate_index], theta[1, covariate_index]]) # 0 or 1
        prob[t] = int(action[t] == best_action)   # 현재 action이 best_action이었다면 1 아니면 0
    
    return action, reward, prob, conds

def causal_thompson(theta, T, all_factors, pObs):
    """
    pObs : 각 arm들을 관찰했을 때 reward를 받았는지 안받았는지 counting한 것
    """
    
    # Initialize
    s = np.array([[1, 1],           # do(X=0),Z=0 에서 보상을 받은 횟수, do(X=1),Z=0 에서 보상을 받은 횟수
                  [1, 1]])          # do(X=0),Z=1 에서 보상을 받은 횟수, do(X=1),Z=1 에서 보상을 받은 횟수

    f = np.array([[1, 1],           
                  [1, 1]])      

    # Calculate pObs from counting : P(Y=1|X=0), P(Y=1|X=1)
    p_Y_X = [pObs[0, 1] / np.sum(pObs[0, :]), pObs[1, 1] / np.sum(pObs[1, :])]      
    
    # Seed distribution : Seed P(y | do(X), z) with observations, whenever X = Z
    s[0, 0] = pObs[0, 1]    # (y=1 | do(X=0), z=0) 인 횟수 = (y=1 | X=0) 인 횟수
    s[1, 1] = pObs[1, 1]    # (y=1 | do(X=1), z=1) 인 횟수 = (y=1 | X=1) 인 횟수
    f[0, 0] = pObs[0, 0]    # (y=0 | do(X=0), z=0) 인 횟수 = (y=0 | X=0) 인 횟수
    f[1, 1] = pObs[1, 0]    # (y=0 | do(X=0), z=1) 인 횟수 = (y=0 | X=1) 인 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    zCount = [0, 0]     # z=0인 횟수, z=1인 횟수

    # Execute one run
    for t in range(T):
        roundFactors = all_factors[:, t]
        B = roundFactors[0]
        D = roundFactors[1]

        # get intuition for trial
        Z = int(roundFactors[2])

        zPrime = 1 - Z              # If z = 0, zPrime = 1; if z = 1, zPrime = 0
        covariateIndex = int(B + D * 2)
        conds[covariateIndex] += 1

        # Compute necessary stats
        zCount[Z] += 1
        
        # intention-specific randomization?
        p_Y_doX_Z = np.array([[s[0, 0] / (s[0, 0] + f[0, 0]), s[0, 1] / (s[0, 1] + f[0, 1])],       
                              [s[1, 0] / (s[1, 0] + f[1, 0]), s[1, 1] / (s[1, 1] + f[1, 1])]])

        # Q1 = E(y_x' | x)  [Counter-intuition] : 현재 B, D로 결정된 X를 따르지 않고 반대로 행동할 경우 얻는 보상
        Q1 = p_Y_doX_Z[Z, zPrime]
        
        # Q2 = E(y_x | x) = P(y=1 | x) [Intuition] : 현재 B, D로 결정된 X를 따라서 행동할 경우 얻는 보상
        Q2 = p_Y_X[Z]
        # Q2 = p_Y_doX_Z[Z, Z]  # 이걸로 해도 동일할듯

        # Perform weighting
        # nan 이 되는 case가 뭐지?
        if np.isnan(abs(Q1 - Q2)):
            bias = 1
        else:
            bias = 1 - abs(Q1 - Q2)

        w = [1, 1]

        # 만약 M1을 해야한다고 가정하자. 지금까지 결과를 봤을 때,
        # M1을 해야했을 때 M2를 했었을 경우 보상을 얻었을 확률이 그냥 M1을 해서 보상을 얻은 확률보다 크다면 (차이가 클수록 bias -> 0)
        # bias를 beta분포에서 얻은 샘플에 곱해서 M1을 당길 확률을 감소시켜줌
        if Q1 > Q2:
            w[Z] = bias
        else:
            w[zPrime] = bias

        # Choose action
        theta_hat = [np.random.beta(s[Z, 0], f[Z, 0]) * w[0], np.random.beta(s[Z, 1], f[Z, 1]) * w[1]]
        chosen_action = np.argmax(theta_hat)
        currentTheta = theta[chosen_action, covariateIndex]

        # Pull lever
        reward_choice = int(np.random.rand() <= currentTheta)

        # Update
        s[Z, chosen_action] += reward_choice
        f[Z, chosen_action] += 1 - reward_choice

        # Record
        action[t] = int(chosen_action == 1)
        reward[t] = reward_choice
        bestAction = np.argmax([theta[0, covariateIndex], theta[1, covariateIndex]])
        prob[t] = int(chosen_action == bestAction)

    return action, reward, prob, conds

def causal_thompson_ns(theta, T, all_factors, pObs):
    
    # Initialize
    s = np.array([[1, 1],       
                  [1, 1]])      
    f = np.array([[1, 1],
                  [1, 1]])      

    # p_X = [np.sum(pObs[0, :]) / np.sum(pObs), np.sum(pObs[1, :]) / np.sum(pObs)]    # 0.5, 0.5
    
    p_Y_X = [pObs[0, 1] / np.sum(pObs[0, :]), pObs[1, 1] / np.sum(pObs[1, :])]      # P(Y=1|X=0), P(Y=1|X=1)
    
    zCount = [0, 0]     # z=0인 횟수, z=1인 횟수

    # # Seed distributiohn : Seed P(y | do(X), z) with observations, whenever X = Z
    # s[0, 0] = pObs[0, 1]    # P(y=1 | do(X=0), z=0) = P(y=1 | X=0)
    # s[1, 1] = pObs[1, 1]    # P(y=1 | do(X=1), z=1) = P(y=1 | X=1)
    # f[0, 0] = pObs[0, 0]    # P(y=0 | do(X=0), z=0) = P(y=0 | X=0)
    # f[1, 1] = pObs[1, 0]    # P(y=0 | do(X=0), z=1) = P(y=0 | X=1)

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary

    # Execute one run
    for t in range(T):
        roundFactors = all_factors[:, t]
        B = roundFactors[0]
        D = roundFactors[1]

        # get intuition for trial
        Z = int(roundFactors[2])

        zPrime = 1 - Z              # If z = 0, zPrime = 1; if z = 1, zPrime = 0
        covariateIndex = int(B + D * 2)
        conds[covariateIndex] += 1

        # Compute necessary stats
        zCount[Z] += 1
        # p_Z = [zCount[0] / np.sum(zCount), zCount[1] / np.sum(zCount)]  #P(Z=0), P(Z=1)
        p_Y_doX_Z = np.array([[s[0, 0] / (s[0, 0] + f[0, 0]), s[0, 1] / (s[0, 1] + f[0, 1])],       # P(y|)
                              [s[1, 0] / (s[1, 0] + f[1, 0]), s[1, 1] / (s[1, 1] + f[1, 1])]])

        # Q1 = E(y_x' | x)  [Counter-intuition] : 현재 B, D로 결정된 X를 따르지 않고 반대로 행동할 경우 얻는 보상
        Q1 = p_Y_doX_Z[Z, zPrime]
        # Q2 = E(y_x | x) = P(y=1 | x) [Intuition] : 현재 B, D로 결정된 X를 따라서 행동할 경우 얻는 보상
        Q2 = p_Y_X[Z]

        # Perform weighting
        bias = abs(Q1 - Q2)
        w = [1, 1]
        if np.isnan(bias):
            weighting = 1
        else:
            weighting = 1 - bias
        if Q1 > Q2:
            w[Z] = weighting
        else:
            w[zPrime] = weighting

        # Choose action
        theta_hat = [np.random.beta(s[Z, 0], f[Z, 0]) * w[0], np.random.beta(s[Z, 1], f[Z, 1]) * w[1]]
        chosen_action = np.argmax(theta_hat)
        currentTheta = theta[chosen_action, covariateIndex]

        # Pull lever
        reward_choice = int(np.random.rand() <= currentTheta)

        # Update
        s[Z, chosen_action] += reward_choice
        f[Z, chosen_action] += 1 - reward_choice

        # Record
        action[t] = int(chosen_action == 1)
        reward[t] = reward_choice
        bestAction = np.argmax([theta[0, covariateIndex], theta[1, covariateIndex]])
        prob[t] = int(chosen_action == bestAction)

    return action, reward, prob, conds

def z_thompson(theta, T, all_factors, pObs):
    """
    maximize based on P(y|do(X),Z) using a new context variable Z
    """
    s = np.array([[1, 1],       # do(X=0),Z=0 일때 보상을 받은 횟수, do(X=1),Z=0 일때 보상을 받은 횟수
                  [1, 1]])      # do(X=0),Z=1 일때 보상을 받은 횟수, do(X=1),Z=1 일때 보상을 받은 횟수
    
    f = np.array([[1, 1], 
                  [1, 1]])

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary

   # Execute one run
    for t in range(T):
        roundFactors = all_factors[:, t]
        B = roundFactors[0]
        D = roundFactors[1]
        Z = int(roundFactors[2])
        covariateIndex = int(B + D * 2)
        conds[covariateIndex] += 1

        # Choose action
        theta_hat = [np.random.beta(s[Z, 0], f[Z, 0]), np.random.beta(s[Z, 1], f[Z, 1])]
        chosen_action = np.argmax(theta_hat)
        currentTheta = theta[chosen_action, covariateIndex]

        # Pull lever
        reward_choice = int(np.random.rand() <= currentTheta)

        # Update
        s[Z, chosen_action] += reward_choice
        f[Z, chosen_action] += 1 - reward_choice

        # Record
        action[t] = int(chosen_action == 1)
        reward[t] = reward_choice
        bestAction = np.argmax([theta[0, covariateIndex], theta[1, covariateIndex]])
        prob[t] = int(chosen_action == bestAction)

    return action, reward, prob, conds

def causal_thompson_sm1(theta, T, all_factors, pObs):
    """
    pObs : 각 arm들을 관찰했을 때 reward를 받았는지 안받았는지 counting한 것
    """
    
    # Initialize
    s = np.array([[1, 1],           # do(X=0),Z=0 에서 보상을 받은 횟수, do(X=1),Z=0 에서 보상을 받은 횟수
                  [1, 1]])          # do(X=0),Z=1 에서 보상을 받은 횟수, do(X=1),Z=1 에서 보상을 받은 횟수

    f = np.array([[1, 1],           
                  [1, 1]])      

    # Calculate pObs from counting : P(Y=1|X=0), P(Y=1|X=1)
    p_Y_X = [pObs[0, 1] / np.sum(pObs[0, :]), pObs[1, 1] / np.sum(pObs[1, :])]      
    
    # Seed distribution : Seed P(y | do(X), z) with observations, whenever X = Z
    s[0, 0] = pObs[0, 1]    # (y=1 | do(X=0), z=0) 인 횟수 = (y=1 | X=0) 인 횟수
    s[1, 1] = pObs[1, 1]    # (y=1 | do(X=1), z=1) 인 횟수 = (y=1 | X=1) 인 횟수
    f[0, 0] = pObs[0, 0]    # (y=0 | do(X=0), z=0) 인 횟수 = (y=0 | X=0) 인 횟수
    f[1, 1] = pObs[1, 0]    # (y=0 | do(X=0), z=1) 인 횟수 = (y=0 | X=1) 인 횟수

    action = np.zeros(T, dtype=int)
    reward = np.zeros(T, dtype=int)
    prob = np.zeros(T, dtype=int)
    conds = np.zeros(4, dtype=int)  # B, D binary
    
    zCount = [0, 0]     # z=0인 횟수, z=1인 횟수

    # Execute one run
    for t in range(T):
        roundFactors = all_factors[:, t]
        B = roundFactors[0]
        D = roundFactors[1]

        # get intuition for trial
        Z = int(roundFactors[2])

        zPrime = 1 - Z              # If z = 0, zPrime = 1; if z = 1, zPrime = 0
        covariateIndex = int(B + D * 2)
        conds[covariateIndex] += 1

        # Compute necessary stats
        zCount[Z] += 1
        
        # intention-specific randomization?
        p_Y_doX_Z = np.array([[s[0, 0] / (s[0, 0] + f[0, 0]), s[0, 1] / (s[0, 1] + f[0, 1])],       
                              [s[1, 0] / (s[1, 0] + f[1, 0]), s[1, 1] / (s[1, 1] + f[1, 1])]])

        # Q1 = E(y_x' | x)  [Counter-intuition] : 현재 B, D로 결정된 X를 따르지 않고 반대로 행동할 경우 얻는 보상
        Q1 = p_Y_doX_Z[Z, zPrime]
        
        # Q2 = E(y_x | x) = P(y=1 | x) [Intuition] : 현재 B, D로 결정된 X를 따라서 행동할 경우 얻는 보상
        Q2 = p_Y_X[Z]

        # # Perform weighting
        # if np.isnan(abs(Q1 - Q2)):
        #     bias = 1
        # else:
        #     bias = 1 - abs(Q1 - Q2)

        # w = [1, 1]

        # # 만약 M1을 해야한다고 가정하자. 지금까지 결과를 봤을 때,
        # # M1을 해야했을 때 M2를 했었을 경우 보상을 얻었을 확률이 그냥 M1을 해서 보상을 얻은 확률보다 크다면 (차이가 클수록 bias -> 0)
        # # bias를 beta분포에서 얻은 샘플에 곱해서 M1을 당길 확률을 감소시켜줌
        # if Q1 > Q2:
        #     w[Z] = bias
        # else:
        #     w[zPrime] = bias

        KL_Q1Q2 = Q1*np.log(Q1/Q2)+(1-Q1)*np.log((1-Q1)/(1-Q2))
        KL_Q2Q1 = Q2*np.log(Q2/Q1)+(1-Q2)*np.log((1-Q2)/(1-Q1))
        # Choose action
        # theta_hat = [np.random.beta(s[Z, 0], f[Z, 0]) * w[0], np.random.beta(s[Z, 1], f[Z, 1]) * w[1]]
        theta_hat = [np.random.beta(s[Z, 0], f[Z, 0]) * 1/Q1, np.random.beta(s[Z, 1], f[Z, 1]) * 1/Q2]
        chosen_action = np.argmax(theta_hat)
        currentTheta = theta[chosen_action, covariateIndex]

        # Pull lever
        reward_choice = int(np.random.rand() <= currentTheta)

        # Update
        s[Z, chosen_action] += reward_choice
        f[Z, chosen_action] += 1 - reward_choice

        # Record
        action[t] = int(chosen_action == 1)
        reward[t] = reward_choice
        bestAction = np.argmax([theta[0, covariateIndex], theta[1, covariateIndex]])
        prob[t] = int(chosen_action == bestAction)

    return action, reward, prob, conds

if __name__=="__main__":

    # Define the parameters and data
    # P(y|do(X), B, D)
    theta = np.array([[0.1, 0.5, 0.4, 0.2],     # X = M1
                    [0.5, 0.1, 0.2, 0.4]])      # X = M2
    
    theta = np.array([[0.4, 0.8, 0.7, 0.5],     # X = M1
                    [0.8, 0.4, 0.5, 0.7]])      # X = M2

    # theta = np.array([[0.4, 0.3, 0.3, 0.4],     # X = M1
    #                 [0.6, 0.1, 0.2, 0.6]])      # X = M2


    T = 1000    # Number of time steps
    N = 1000    # Number of Monte Carlo Samples
    
    N_obs = 200 # Number of observational samples, evenly divided per arm. : 각 arm의 결과를 관찰한 횟수

    # P(y|X) : [[P(y=0|X=0), P(y=1|X=0)],
    #          [P(y=0|X=1), P(y=1|X=1)]]
    p_obs = np.array([[(1-theta[0,0] + 1-theta[0,3]) * N_obs/4, (theta[0,0] + theta[0,3]) * N_obs/4],     # [85, 15]
                      [(1-theta[1,1] + 1-theta[1,2]) * N_obs/4, (theta[1,1] + theta[1,2]) * N_obs/4]])    # [85, 15]

    # Total number of algorithms
    # algorithms = ['epsilon_greedy', 'UCB', 'observational', 'coin_flipping', 'thompson']
    # algorithms = ['causal_thompson', 'causal_thompson_ns', 'z_thompson']
    algorithms = ['causal_thompson', 'thompson', 'z_thompson']
    # algorithms = ['epsilon_greedy', 'UCB', 'observational', 'coin_flipping', 'thompson', 'z_thompson', 'causal_thompson']
    # algorithms = ['causal_thompson', 'causal_thompson_sm1', 'z_thompson']
    n_algs = len(algorithms)

    # Arrays to store results
    action_sum = np.zeros((n_algs, T))
    reward_sum = np.zeros((n_algs, T))
    prob_sum = np.zeros((n_algs, T))
    cond_sum = np.zeros((n_algs, 4))  # B, D binary

    # Plot settings
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    N_PLOTS = 2

    # Open figure
    plt.figure(figsize=(14, 5))

    for alg in range(n_algs):
        # Get handle of algorithm to run
        print(f'Running algorithm: {algorithms[alg]}')
        current_factors = np.zeros((3, T, N))

        # Generate Monte Carlo simulations
        for n in range(N):
            # Determine covariates for this run
            for t in range(T):
                B = np.random.rand() <= 0.5
                D = np.random.rand() <= 0.5
                Z = np.logical_xor(B, D).astype(int)
                current_factors[:, t, n] = [B, D, Z]

            # T step까지의 하나의 run을 얻어냄
            action, reward, prob, conds = locals()[algorithms[alg]](theta, T, current_factors[:, :, n], p_obs)
            
            # Collect stats : 각 알고리즘 별 크기는 T. cond는 4
            # 하나의 run에 대한 결과를 저장 -> N번 저장해야함
            action_sum[alg, :] += action
            reward_sum[alg, :] += reward
            prob_sum[alg, :] += prob
            cond_sum[alg, :] += conds

            # Report progress
            if n % 100 == 0:
                print(f' Samples: {n}')

        # Monte Carlo estimates -> N개의 T step run들의 결과를 이용해서 p_action, regret을 평균으로 계산
        
        # N개의 prob를 평균 낸 것 -> 즉, 평균적으로 time step t에서 최적의 행동을 하는 확률을 계산한 것
        p_action = prob_sum[alg, :] / N   
        
        # np.max(theta[: ,i]) : ith covariate index에서 가장 최적의 선택
        # (cond_sum[alg, 0] / N) : N번의 simulation에서 ith covariate index가 나온 횟수
        # 따라서 이론적인 reward 최댓값을 구하고, uniform하게 T step으로 분배? 하고 누적 reward_sum을 빼서 T에 따른 누적 regret을 구한 것
        regret = (np.max(theta[:, 0]) * (cond_sum[alg, 0] / N) +
                np.max(theta[:, 1]) * (cond_sum[alg, 1] / N) +
                np.max(theta[:, 2]) * (cond_sum[alg, 2] / N) +
                np.max(theta[:, 3]) * (cond_sum[alg, 3] / N)) * np.arange(1, T+1) / T - np.cumsum(reward_sum[alg, :] / N)

        # Probability of Optimal Action Plot
        plt.subplot(1, N_PLOTS, 1)
        plt.plot(p_action, color=colors[alg], label=algorithms[alg])
        plt.title('Probability of Optimal Action')
        plt.xlabel('Trial')
        plt.ylabel('Probability')
        plt.ylim([0.4, 1.0])
        
        # Cum.regret Plot
        plt.subplot(1, N_PLOTS, 2)
        plt.plot(regret, color=colors[alg], label=algorithms[alg])
        plt.title('Regret')
        plt.xlabel('Trial')
        plt.ylabel('Cum. Regret')
        plt.ylim([0, 25])

    plt.legend(loc='upper left')
    plt.show()