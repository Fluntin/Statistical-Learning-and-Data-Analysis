results_df_Moment_Estimators, results_df_Gradient_Descent, results_df_Newton_Raphson

#Get Data
initial_bayesian_df_tricks = all_tricks_df_with_zeroes.copy()[all_tricks_df_with_zeroes['id'].isin(contestants)]

initial_bayesian_df_tricks['theta_average'] = round(results_df_Moment_Estimators['theta_average'],4)

initial_bayesian_df_tricks['alpha_trick_Moment'] = results_df_Moment_Estimators['alpha_trick']
initial_bayesian_df_tricks['beta_trick_Moment'] = results_df_Moment_Estimators['beta_trick']

initial_bayesian_df_tricks['alpha_trick_Gradient'] = results_df_Gradient_Descent['alpha_trick']
initial_bayesian_df_tricks['beta_trick_Gradient'] = results_df_Gradient_Descent['beta_trick']

initial_bayesian_df_tricks['alpha_trick_Gauss'] = results_df_Newton_Raphson['alpha_trick']
initial_bayesian_df_tricks['beta_trick_Gauss'] = results_df_Newton_Raphson['beta_trick']

merged_tricks_with_zeroes = [item for sublist in initial_bayesian_df_tricks['tricks'] for item in sublist]

merged_tricks_without_zeroes=[x for x in merged_tricks_with_zeroes if x != 0]

initial_guess_trick_with_zeroes=get_moment_estimators(merged_tricks_with_zeroes)
inital_guess_tricks_without_zeroes=get_moment_estimators(merged_tricks_without_zeroes)

print("Without Zeroes the intial guess for Alpha and Beta via moment estimators would be:", inital_guess_tricks_without_zeroes)
print("With Zeroes the intial guess for Alpha and Beta via moment estimators would be:", initial_guess_trick_with_zeroes)

initial_bayesian_df_tricks = initial_bayesian_df_tricks.assign(
    population_alpha=inital_guess_tricks_without_zeroes[0],
    population_beta=inital_guess_tricks_without_zeroes[1],
    population_alpha0=initial_guess_trick_with_zeroes[0],
    population_beta0=initial_guess_trick_with_zeroes[1]
)

def log_prior(alpha, beta, precision, lambda_hyper):
    theta_hyper = lambda_hyper*precision
    return   theta_hyper * np.log(lambda_hyper) - loggamma(theta_hyper) + (theta_hyper - 1) * np.log(alpha + beta + 1) - lambda_hyper * (alpha + beta + 1) - np.log(alpha + beta)


def log_posterior(alpha, beta, theta, data):
    precision = 5
    lambda_hyper = 0.5

    log_p = log_prior(alpha, beta, precision, lambda_hyper)

    for z_i in data:
        
        if z_i == 0:
            log_p += np.log(1 - theta)
        else:
            log_p += np.log(theta) + loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta) + (alpha - 1) * np.log(z_i) + (beta-1) * np.log(1 - z_i)

    return log_p

def metropolis_algorithm(data, initial_guess, number_of_samples):
    
    alphas = np.zeros((number_of_samples))
    betas = np.zeros((number_of_samples))
    thetas = np.ones((number_of_samples))

    thetas [0] = initial_guess[0]
    alphas[0] = initial_guess[1] 
    betas[0] = initial_guess[2]
    
    
    for i in range(number_of_samples - 1):
        last_alpha = alphas[i]
        last_beta = betas[i]
        last_theta = thetas[i]
        
       # Att exponentiera ser till att alpha och beta är positivt, hade abs innan men det kan leda till alpha och beta med bias och det gick snabbare att konvergera
        proposal_alpha =  np.exp(np.log(last_alpha) + stats.norm.rvs( 0.5, size=1))[0]
        proposal_beta =   np.exp(np.log(last_beta) + stats.norm.rvs(0.5, size=1))[0]
        
        proposal_theta = abs(stats.uniform.rvs(0,1, size = 1))[0]

        
        # Note that the acceptance probability rho is calculated for the *pair* of
        # proposed samples.
        log_rho = log_posterior(proposal_alpha, proposal_beta, proposal_theta, data) - log_posterior(last_alpha, last_beta, last_theta, data)
        
        u = stats.uniform.rvs()
        
        if np.log(u) <= log_rho:
            alphas[i + 1] = proposal_alpha
            betas[i + 1] = proposal_beta
            thetas[i+1] = proposal_theta
        else:
            alphas[i + 1] = last_alpha
            betas[i + 1] = last_beta
            thetas[i +1] = last_theta
    
    return alphas, betas, thetas

import pandas as pd
import numpy as np
from scipy import stats

# Your metropolis_algorithm function...

# Convert your DataFrame to a list of dictionaries
data_list = initial_bayesian_df_tricks.to_dict(orient='records')

# Create an empty list to store the results
results = []

for skateboarder in data_list:
    tricks = skateboarder['tricks']
    
    # Extracting initial guesses
    initial_guesses = [
        [skateboarder['theta_average'], skateboarder['alpha_trick_Moment'], skateboarder['beta_trick_Moment']],
        [skateboarder['theta_average'], skateboarder['alpha_trick_Gradient'], skateboarder['beta_trick_Gradient']],
        [skateboarder['theta_average'], skateboarder['alpha_trick_Gauss'], skateboarder['beta_trick_Gauss']],
        [skateboarder['theta_average'], skateboarder['population_alpha'], skateboarder['population_beta']],
        [skateboarder['theta_average'], skateboarder['population_alpha0'], skateboarder['population_beta0']]
    ]
    
    number_of_samples=15000
    # Run the Metropolis algorithm for each initial guess
    moment_chain = metropolis_algorithm(tricks, initial_guesses[0], number_of_samples)
    gradient_chain = metropolis_algorithm(tricks, initial_guesses[1], number_of_samples)
    gauss_chain = metropolis_algorithm(tricks, initial_guesses[2], number_of_samples)
    population_chain = metropolis_algorithm(tricks, initial_guesses[3], number_of_samples)
    population0_chain = metropolis_algorithm(tricks, initial_guesses[4], number_of_samples)
    
    # Store results in the specified format
    skateboarder['moment_chain'] = moment_chain
    skateboarder['gradient_chain'] = gradient_chain
    skateboarder['gauss_chain'] = gauss_chain
    skateboarder['population_chain'] = population_chain
    skateboarder['population0_chain'] = population0_chain
    
    results.append(skateboarder)

# Convert results back to DataFrame
initial_bayesian_df_tricks = pd.DataFrame(results)

def cumulative_avg(data):
    return np.cumsum(data) / np.arange(1, len(data) + 1)

# Adding new columns to the dataframe before filling them
chains_list = ['moment', 'gradient', 'gauss', 'population', 'population0']
for chain_name in chains_list:
    for param in ['alpha', 'beta', 'theta']:
        initial_bayesian_df_tricks[f'{chain_name}_{param}_cumavg'] = pd.Series(dtype=object)

# Iterate through each row of the DataFrame
for idx, skateboarder in initial_bayesian_df_tricks.iterrows():
    # Extract chains
    chains = [
        skateboarder['moment_chain'],
        skateboarder['gradient_chain'],
        skateboarder['gauss_chain'],
        skateboarder['population_chain'],
        skateboarder['population0_chain']
    ]
    
    # For each chain, compute cumulative averages and save to new columns
    for chain_name, chain in zip(chains_list, chains):
        alpha_values, beta_values, theta_values = chain
        initial_bayesian_df_tricks.at[idx, f'{chain_name}_alpha_cumavg'] = list(cumulative_avg(alpha_values))
        initial_bayesian_df_tricks.at[idx, f'{chain_name}_beta_cumavg'] = list(cumulative_avg(beta_values))
        initial_bayesian_df_tricks.at[idx, f'{chain_name}_theta_cumavg'] = list(cumulative_avg(theta_values))
        
        import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Seaborn is great for styling and color palettes

# Use a Seaborn palette for more advanced colors
colors = sns.color_palette("deep", 5)

# Adjust column count to 4 to add space for scatter plots
fig, axes = plt.subplots(nrows=len(initial_bayesian_df_tricks), ncols=4, figsize=(20, 4 * len(initial_bayesian_df_tricks)))

for idx, skateboarder in initial_bayesian_df_tricks.iterrows():
    # Extract chains
    chains = [
        skateboarder['moment_chain'],
        skateboarder['gradient_chain'],
        skateboarder['gauss_chain'],
        skateboarder['population_chain'],
        skateboarder['population0_chain']
    ]
    
    # Extract and plot values
    for chain, color in zip(chains, colors):
        alpha_values, beta_values, theta_values = chain
        
        # Plot cumulative average of alphas
        axes[idx, 0].plot(np.cumsum(alpha_values) / np.arange(1, len(alpha_values) + 1), color=color, linewidth=1.5)
        axes[idx, 0].set_title(f"{skateboarder['id']} - Alphas")
        axes[idx, 0].grid(True, which="both", linestyle='--', linewidth=0.5)
        
        # Plot cumulative average of betas
        axes[idx, 1].plot(np.cumsum(beta_values) / np.arange(1, len(beta_values) + 1), color=color, linewidth=1.5)
        axes[idx, 1].set_title(f"{skateboarder['id']} - Betas")
        axes[idx, 1].grid(True, which="both", linestyle='--', linewidth=0.5)
        
        # Plot cumulative average of thetas
        axes[idx, 2].plot(np.cumsum(theta_values) / np.arange(1, len(theta_values) + 1), color=color, linewidth=1.5)
        axes[idx, 2].set_title(f"{skateboarder['id']} - Thetas")
        axes[idx, 2].grid(True, which="both", linestyle='--', linewidth=0.5)
        
        # Scatter plot for alphas and betas in the fourth column
        axes[idx, 3].scatter(alpha_values, beta_values, color=color, s=10, alpha=0.5)
        axes[idx, 3].set_title(f"{skateboarder['id']} - Alphas vs Betas")
        axes[idx, 3].set_xlabel('Alphas')
        axes[idx, 3].set_ylabel('Betas')
        axes[idx, 3].grid(True, which="both", linestyle='--', linewidth=0.5)

for ax_row in axes:
    for ax in ax_row:
        sns.despine(ax=ax)  # Remove top and right spines for a cleaner look
        ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks

# Tight layout
plt.tight_layout()
plt.show()

initial_bayesian_df_tricks['alpha_bayes_X_mean'] = np.nan
initial_bayesian_df_tricks['beta_bayes_X_mean'] = np.nan
initial_bayesian_df_tricks['theta_bayes_mean'] = np.nan
initial_bayesian_df_tricks['alpha_bayes_X_s^2'] = np.nan
initial_bayesian_df_tricks['beta_bayes_X_s^2'] = np.nan
initial_bayesian_df_tricks['theta_bayes_s^2'] = np.nan

chains = ['moment_chain', 'gradient_chain', 'gauss_chain', 'population_chain', 'population0_chain']

for idx, skateboarder in initial_bayesian_df_tricks.iterrows():
    
    # Combine from all chains
    all_alphas = np.concatenate([skateboarder[chain][0] for chain in chains])
    all_betas = np.concatenate([skateboarder[chain][1] for chain in chains])
    all_thetas = np.concatenate([skateboarder[chain][2] for chain in chains])

    # Update the dataframe
    initial_bayesian_df_tricks.at[idx, 'alpha_bayes_X_mean'] = np.mean(all_alphas)
    initial_bayesian_df_tricks.at[idx, 'beta_bayes_X_mean'] = np.mean(all_betas)
    initial_bayesian_df_tricks.at[idx, 'theta_bayes_mean'] = np.mean(all_thetas)
    
    initial_bayesian_df_tricks.at[idx, 'alpha_bayes_X_s^2'] = np.var(all_alphas, ddof=1)
    initial_bayesian_df_tricks.at[idx, 'beta_bayes_X_s^2'] = np.var(all_betas, ddof=1)
    initial_bayesian_df_tricks.at[idx, 'theta_bayes_s^2'] = np.var(all_thetas, ddof=1)

initial_bayesian_df_tricks[['id', "alpha_bayes_X_mean", "beta_bayes_X_mean", "theta_bayes_mean", "alpha_bayes_X_s^2", "beta_bayes_X_s^2", "theta_bayes_s^2"]]



