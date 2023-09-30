def aggregate_tricks(tricks):
    # This function aggregates the tricks by filtering out zero values
    return list(tricks[tricks != 0])

# Assuming the tricks are stored in columns named 'trick 1', 'trick 2', etc.
trick_columns = ['trick 1', 'trick 2', 'trick 3', 'trick 4']  # Add more columns if needed

# Group by 'id' and aggregate tricks for each skateboarder
all_tricks_df = df.melt(id_vars='id', value_vars=trick_columns)\
                  .groupby('id')['value']\
                  .agg(aggregate_tricks)\
                  .reset_index()

# Rename the columns for clarity
all_tricks_df.columns = ['id', 'tricks']

# Print the result
print(all_tricks_df)



def gradient(alpha_beta, data):
    alpha, beta = alpha_beta
    n = len(data)
    
    psi_alpha_plus_beta = sp.psi(alpha + beta)
    psi_alpha = sp.psi(alpha)
    psi_beta = sp.psi(beta)
    
    grad_alpha = n * psi_alpha_plus_beta - n * psi_alpha + np.sum(np.log(data))
    grad_beta = n * psi_alpha_plus_beta - n * psi_beta + np.sum(np.log(1 - data))
    
    return np.array([grad_alpha, grad_beta])

def graddes(initialization, stepsize, num_iter, data):
    thetas = [initialization]
    
    for i in range(num_iter):
        #thetas.append(thetas[-1] - stepsize * gradient(thetas[-1], data))
        thetas = thetas + [thetas[-1] - (stepsize)*gradient(thetas[-1], data)]
        
    return thetas

def gradient_method(samps):
    alpha_init = samps.mean()
    beta_init = samps.std()
    
    initialization = np.array([alpha_init, beta_init])
    G = graddes(initialization, 0.001, 1000, samps)
    #Garray = np.array([list(x) for x in G])
    #plt.plot(Garray[:,0], Garray[:,1], 'o-r', markersize = 3)
    #plt.xlabel('$α_k$', fontsize=14)
    #plt.ylabel('$β_k$', fontsize=14)
    #plt.show()
    
    return G

results=[]
for index, row in all_tricks_df.iterrows():
    samps = np.array(row['tricks'])
    alpha_beta = gradient_method(samps)
    print(alpha_beta[-1])
    results.append(alpha_beta[-1])

print(len(results))
