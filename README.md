# Statistical Learning and Data Analysis, Autumn 2023 Project 

## 1. Introduction

This project aims to build a model to predict the competitors in the season finale of a particular sports competition, namely the 2022 Street League Skateboarding (SLS) Super Crown Championship. The event consists of a last-chance qualifying competition (Last Chance Qualifier or LCQ) and the final. There are eight spots in the final. Four skateboarders have already qualified based on their performance throughout the season. These, given by last names, are:

$
\text { Horigome Joslin Milou Ribeiro G. }
$

The LCQ has sixteen competitors, and the four skateboarders with the highest scores win the remaining four spots in the final. The skateboarders competing in the LCQ (given by last names) are:

$\begin{array}{cccc}\text { Majerus } & \text { Oliveira } & \text { Decenzo } & \text { Santiago } \\ \text { Papa } & \text { Eaton } & \text { Mota } & \text { Shirai } \\ \text { Jordan } & \text { Hoefler } & \text { Hoban } & \text { Gustavo } \\ \text { Ribeiro C } & \text { O'neill } & \text { Foy } & \text { Midler. }\end{array}$

In the LCQ, each skateboarder gets two runs (consisting of 45 seconds to perform as many tricks as possible) and four trick attempts (single trick attempts) where the skateboarder attempts to do just one (difficult) trick. A score between 0 and 10 is assigned to each run and each trick. A trick gets a score of 0 only if the skateboarder fails to land the trick. The total score for a skateboarder's performance is calculated as the sum of their two highest scores on tricks and their highest score on a run. The goal of this project is to build a model to predict these four skateboarders based on data collected during the 2022 season, which consisted of three other events.

Each of the three events for which we have data consists of a qualifying and a final round. In each of these two competitions, skateboarders are assigned an order in which they take turns with the different activities (runs and tricks). The activities are conducted in the following order: run 1, run 2, trick 1, trick 2, trick 3, trick 4. All competitors perform run 1 first, then run 2, then trick 1, and so on. Skateboarders perform each activity in the same order. For example, if we have three competitors, we assign them an order $S_{1}, S_{2}, S_{3}$. The competition begins with run 1 for $S_{1}$, then run 1 for $S_{2}$, then run 1 for $S_{3}$, then run 2 for $S_{1}$, and so on.

The qualifying competitions have the same format as the LCQ except that skateboarders are divided into different heats in which they perform the activities in an order determined by the competitors in that heat. The final consists of the top 8 skateboarders with the highest total scores from the corresponding qualifying competition. In the finals, each skateboarder first gets two runs and four tricks. Then, their total scores are calculated. The four skateboarders with the lowest total scores are eliminated. Afterward, the remaining four skateboarders each get two more tricks. When all additional tricks are completed, the total scores are recalculated. The skateboarder with the highest total score wins!

The data is in the form of a CSV file called SLS22.csv. The CSV file contains the following columns:

| Name | Data Type | Description |
| :--- | :--- | :--- |
| id | string | skateboarder's last name |
| location | string | location of the competition |
| month | integer | competition month |
| year | integer | competition year |
| comp | string | competition type ('prelim men' or 'final men') |
| heat | integer | Heat in which the skateboarder participated (heat=1 if comp = 'final men') |
| run 1 | float | score for run 1 |
| run 2 | float | score for run 2 |
| trick 1 | float | score for trick 1 |
| trick 2 | float | score for trick 2 |
| trick 3 | float | score for trick 3 |
| trick 4 | float | score for trick 4 |
| trick 5 | float | score for trick 5 ('NaN' if eliminated after trick 4) |
| trick 6 | float | score for trick 6 ('NaN' if eliminated after trick 4) |

Here's the translation of the text into English:

## 2. Tasks

Solve the following tasks and record your solutions in a PDF file in the form of a report. Along with the report, you must also submit your code in the form of a Python script or a Jupyter notebook with proper markdown. You may work on the project with one (and only one) other student in the course, but you must write your own code and your own report. You may not submit two identical reports! The report and code should be written in your own words to demonstrate your understanding of the solutions. You should also include the name of the person you worked with on the project on the first page of the report.

1. Warm-up. The following tasks should acquaint you with the dataset and prepare the data for use when building your predictive models.

   (a) All scores in the dataset are currently numbers between 0 and 10. Normalize these values in the dataset so that they range from 0 to 1.

   (b) Create a histogram for all trick scores for tricks 1-4. What do you observe? Is there a certain value that appears more frequently than the others? If so, how does this value compare to the others?

   (c) For each trick 1-4, create a new column named 'make i' for $i=1,2,3,4$ so that the value of 'make i' in a given row is 1 if the skateboarder landed trick $i$, and 0 otherwise.

   (d) For each skateboarder, estimate the probability that a trick receives a score greater than 0.6, given that the skateboarder landed the trick. What is the probability that the skateboarder fails to land a specific trick? What do you observe? Relate your observations to those in part (b).

   (e) Create a scatter plot for run score 1 against run score 2. Do you see any clear correlation from the plot?

2. A frequentist model. We would like to build a model that can predict which of the 16 skateboarders in the LCQ will earn a spot in the final. One way to do this is to build a model for each skateboarder, use the models to simulate run scores and trick scores for each skateboarder, and combine the simulations to simulate the LCQ. We can simulate multiple LCQs and extract the four skateboarders with the highest total scores from each one. Our prediction will be the mode of these results. Note that this model assumes that the skateboarders' performances are independent. For simplicity, we assume that the score on a particular run $Y_{i}$ and the score on a particular trick $X_{i}$ are independent for each skateboarder $i$. We also assume that all trick scores and run scores are independent and identically distributed outcomes from $X_{i}$ and $Y_{i}$. We can start by specifying a model for $X_{i}$ and $Y_{i}$ based on the observations in Task 1:

   $ X_{i}= \begin{cases}0 & \text { if } V_{i}=0, \\ Z_{i} & \text { if } V_{i}=1, \end{cases} $

   where $V_{i} \sim \text{Ber}(\theta_{i})$, $Z_{i} \sim \text{Beta}(\alpha_{i}, \beta_{i})$, and $V_{i} \perp Z_{i}$. It can be shown that

   $ f_{X_{i}}(x_{i} | \theta_{i}, \alpha_{i}, \beta_{i}) = (1-\theta_{i}) \mathbf{1}_{x_{i}=0}+\theta_{i} f_{Z_{i}}(z_{i}). $

   (a) Provide a point estimate for each $\theta_{i}$, the probability that skateboarder $i$ lands a trick.

   (b) Provide a point estimate for the parameters $\left[\alpha_{i}, \beta_{i}\right]^{\mathrm{T}}$ for each skateboarder $i$. Are there skateboarders for whom your chosen point estimate does not exist? If so, suggest an alternative point estimate for these $\theta_{i}$. Justify your choices of point estimates.

   (c) Propose a model for $Y_{i}$ and provide a point estimate for your model's parameters. Justify your choices for the model and point estimates.

   (d) Use your model for $\left[X_{i}, Y_{i}\right]^{\mathrm{T}}$ to simulate 5000 LCQs, and for each simulation, extract the four skateboarders $\boldsymbol{W}=\left[W_{1}, W_{2}, W_{3}, W_{4}\right]^{\mathrm{T}}$ with the highest total scores. What is the mode of $\boldsymbol{W}_{1}, \ldots, \boldsymbol{W}_{5000}$? The actual winners of the LCQ are Gustavo, Hoban, Eaton, and Decenzo.

   How many of the actual winners are predicted by the mode? What is the estimated probability of the actual winners based on your simulations? Based on the mode?

3. A Bayesian model. As an alternative to the frequentist model developed in Task 2, we can consider a Bayesian model.

   (a) Propose a joint prior distribution for the parameters $\left[\Theta_{i}, A_{i}, B_{i}\right]^{\mathrm{T}}$ for $X_{i}$, assuming $\Theta_{i} \perp A_{i}, B_{i}$ for all $i$. Justify your choice.

   (b) Generate 5000 random samples from the posterior distribution

   $ f_{\theta_{i}, \alpha_{i}, \beta_{i} | \boldsymbol{X}_{i}}(\theta_{i}, \alpha_{i}, \beta_{i} | \boldsymbol{x}_{i}). $

   Plot your resulting samples for the marginal posterior distributions:

   $ f_{\theta_{i} | \boldsymbol{X}_{i}}(\theta_{i} | \boldsymbol{x}_{i}) $

   and

   $ f_{\alpha_{i}, \beta_{i} | \boldsymbol{X}_{i}}(\alpha_{i}, \beta_{i} | \boldsymbol{x}_{i}). $

   Calculate the posterior sample mean and posterior sample variance for each parameter $\theta_{i}$, $\alpha_{i}$, and $\beta_{i}$ for all skateboarders.

   (c) Propose a (simultaneous) prior distribution for the parameters of your model for $Y_{i}$ from Task 2(c) and justify your choice. Assume that the model parameters for skateboarder $i$ are independent of all other parameters, including $\theta_{i}$, $\alpha_{i}$, and $\beta_{i}$. Generate 5000 samples from the posterior distribution (be sure to save these samples!) and create a scatter plot of the results. What are the sample means and sample variances for each of your parameters based on your samples?

   (d) Use your Bayesian model for $\left[X_{i}, Y_{i}\

right]^{T}$ to simulate 5000 LCQs by drawing samples from the appropriate posterior predictive distributions. What is the mode of your results $\boldsymbol{W}_{1}, \ldots, \boldsymbol{W}_{5000}$? How many of the actual winners are predicted? What is the estimated probability of the actual winners based on your results? Based on the mode?

   (e) In the model in Task 3(d), we assumed that the parameters $\boldsymbol{\Upsilon}_{i}$ for $Y_{i}$ and the parameters $\boldsymbol{\Theta}_{i}=\left[\Theta_{i}, A_{i}, B_{i}\right]^{T}$ for $X_{i}$ are independent given data (why?). At the same time, we did not assume that $\Theta_{i} \perp A_{i}, B_{i}$ are independent given data. Let $X_{i}^{(1)}, X_{i}^{(2)}, X_{i}^{(3)}, X_{i}^{(4)}$ denote skateboarder $i$'s four trick scores, let $Y_{i}^{(1)}, Y_{i}^{(2)}$ denote skateboarder $i$'s two run scores, and let $O_{i}$ denote their total score. Draw a directed acyclic graph with as few edges as possible so that the joint distribution for $O_{i}, X_{i}^{(1)}, X_{i}^{(2)}, X_{i}^{(3)}, X_{i}^{(4)}, Y_{i}^{(1)}, Y_{i}^{(2)}, \Theta_{i}, A_{i}, B_{i}$, and $\Upsilon$ is Markov with respect to it. Based on your graph, can you conclude that the marginal posterior distribution for $\Theta_{i}, A_{i}$, and $B_{i}$ factorizes as

   $ f_{\theta_{i}, \alpha_{i}, \beta_{i} | \boldsymbol{X}_{i}}(\theta_{i}, \alpha_{i}, \beta_{i} | \boldsymbol{x}_{i}) = f_{\theta_{i} | \boldsymbol{X}_{i}}(\theta_{i} | \boldsymbol{x}_{i}) f_{\alpha_{i}, \beta_{i} | \boldsymbol{X}_{i}}(\alpha_{i}, \beta_{i} | \boldsymbol{x}_{i})? $

   Considering your parameters $\boldsymbol{\Upsilon}_{i}$ for $Y_{i}$ and the parameters $\boldsymbol{\Theta}_{i}$ for $X_{i}$, does our assumption that

   $ \boldsymbol{\Upsilon}_{i} \perp \boldsymbol{\Theta}_{i} | X_{i}^{(1)}, X_{i}^{(2)}, X_{i}^{(3)}, X_{i}^{(4)}, Y_{i}^{(1)}, Y_{i}^{(2)} $

   make sense? Can we assume the independence $\boldsymbol{\Upsilon}_{i} \perp \boldsymbol{\Theta}_{i} | O_{i}$ if only the data $o_{i}$ is given instead?

4. A Bayesian model with a hierarchy. To account for possible variations in skateboarders' performances across different competitions, we can build a model that uses a hierarchy. As seen in the lectures, we can construct a Bayesian hierarchy for $V_{i} \sim \text{Ber}(\theta_{i})$ by grouping outcomes $v_{i}$ by the different competitions. For simplicity, we use our frequentist point estimates for the parameters $\alpha_{i}, \beta_{i}$ and the parameters for $Y_{i}$ from Task 2.

   (a) Assume that $\Theta_{i} | A_{i}=\alpha_{i}, B_{i}=\beta_{i} \sim \text{Beta}(\alpha_{i}, \beta_{i})$ and choose an appropriate joint prior distribution for $\left[\Theta_{i}, A_{i}, B_{i}\right]^{T}$. Justify your choice.

   (b) Generate 5000 random samples from the joint posterior distribution

   $ f_{A_{i}, B_{i} | \boldsymbol{X}_{i}}(a_{i}, b_{i} | \boldsymbol{x}_{i}). $

   Use your simulations to generate 5000 random samples from the marginal posterior distribution $\Theta_{i} | \boldsymbol{X}_{i}=\boldsymbol{x}_{i}$. Make plots with your samples for the following posterior distributions:

   $ f_{\theta_{i} | \boldsymbol{X}_{i}}(\theta_{i} | \boldsymbol{x}_{i}) $

   and

   $ f_{A_{i}, B_{i} | \boldsymbol{X}_{i}}(a_{i}, b_{i} | \boldsymbol{x}_{i}). $

   Provide estimates for the posterior expected values and posterior variances for each of the parameters. How do these variances for $\theta_{i}$ compare to the variances for $\theta_{i}$ calculated for the model in Task 3?

   (c) Using your 5000 samples from part (b), simulate 5000 LCQ competition winners and calculate the mode of the results. What are the respective estimated probabilities for the actual winners and your mode?

5. Discussion. It is always important to reflect on our model assumptions when conducting statistical inference. Specifically, it is important to assess how the models can be improved.

   (a) How do the results (skateboarders in the modes) of the different models compare? Which skateboarders are correctly predicted, and which are not? Provide some possible explanations for the differences in predictions between the different models. Which model do you prefer, and why?

   (b) How do your estimates for $\theta_{i}$ in Task 1 compare to your estimated means and variances for $\theta_{i}$ in Tasks 3 and 4? What is the expected score for a trick for each skateboarder, given that the trick is successfully landed? What is the expected run score? Considering the skateboarders predicted to win according to the different models, do these statistics provide any insights into successful strategies for winning? (For example, does focusing on a good run score rather than good trick scores work? Are there examples where this strategy succeeds? Is it better to have better trick scores with high variance or slightly worse trick scores with less variance? etc.)

   (c) Estimate the mean and standard deviation of each skateboarder's total score for the models in Task 3 and 4. Does this statistics support your predictions? According to this statistics, what must happen for the results to match the actual winners?

   (d) In all models, we assumed that the skateboarder's performances are independent. For example, we assumed that all $V_{i}$ are independent. Does this seem like a reasonable assumption? Justify your answer.

   (e) In all models, we disregarded the order in which the skateboarders take turns. Does this seem like a reasonable thing to do? Why or why not?