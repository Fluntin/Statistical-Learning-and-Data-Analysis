# SF1930 Statistical Learning and Data Analysis, Fall 2023 Project

## 1. Introduction

This project aims to build a model to predict the competitors in the season finale of a particular sports competition: the 2022 Street League Skateboarding (SLS) Super Crown Championship. The event consists of a last-chance qualifier (LCQ) and the final. There are eight spots in the final. Four skateboarders have already qualified based on their performance throughout the season. These, given by their surnames, are:

```
Horigome Joslin Milou Ribeiro G.
```

The LCQ features sixteen competitors, and the top four scorers secure the remaining four spots in the final. Skateboarders competing in the LCQ (given by surnames) are:

```
Majerus, Oliveira, Decenzo, Santiago, Papa, Eaton, Mota, Shirai, Jordan, Hoefler, Hoban, Gustavo, Ribeiro C, O'neill, Foy, Midler.
```

In the LCQ, each skateboarder gets two runs (each lasting 45 seconds to perform as many tricks as possible) and four single trick attempts. A score between 0 and 10 is assigned to each run and each trick. A trick earns a score of 0 only if the skateboarder fails to land it. The total score for a skateboarder is calculated as the sum of their two highest trick scores and their highest run score. The top four skateboarders by total score claim the four remaining spots in the final. The goal of this project is to build a model to predict these four skateboarders based on data collected during the 2022 season comprising three other events.

Each of the three events we have data for consists of a qualifier and a final. In each of these two competitions, skateboarders are assigned an order in which they rotate through various activities (runs and tricks). Activities are performed in the order: run 1, run 2, trick 1, trick 2, trick 3, and trick 4. All participants first perform run 1, then run 2, then trick 1, and so on. Skateboarders perform each activity in the same order.

Qualifying events have the same format as the LCQ, except that skateboarders are divided into heats where they undertake activities based on the order of competitors in that specific heat. The final consists of the top 8 skateboarders from the respective qualifier. In the finals, each skateboarder first gets two runs and four tricks. Then, their total scores are calculated. The four skateboarders with the lowest total scores are eliminated. The remaining four skateboarders get two more tricks. After all the additional tricks are completed, total scores are recalculated. The skateboarder with the highest total score wins!

The data is in the form of a CSV file named `SLS22.csv`. This file contains the following columns:

```
| Name      | Data Type | Description                                                           |
|-----------|----------|-----------------------------------------------------------------------|
| id        | string   | skateboarder's surname                                                |
| location  | string   | event location                                                        |
| month     | integer  | month of the competition                                               |
| year      | integer  | year of the competition                                                |
| comp      | string   | competition type ('prelim men' or 'final men')                        |
| heat      | integer  | Heat in which the skateboarder participated (heat =1 if comp = 'final men')|
| run 1     | float    | score for run 1                                                       |
| run 2     | float    | score for run 2                                                       |
| trick 1   | float    | score for trick 1                                                     |
| trick 2   | float    | score for trick 2                                                     |
| trick 3   | float    | score for trick 3                                                     |
| trick 4   | float    | score for trick 4                                                     |
| trick 5   | float    | score for trick 5 ('NaN' if eliminated after trick 4)                 |
| trick 6   | float    | score for trick 6 ('NaN' if eliminated after trick 4)                 |
```

You can load the CSV file using the Python package, Pandas. This will load the CSV file into a Pandas dataframe. You can gather basic information about the dataframe using `df.describe()`. You can extract individual columns from the dataframe by indexing it. For instance, the scores for run 1 are provided by the "run 1" column. You can also select certain rows from the dataframe by integer indexing. 

Here's a small Python snippet to guide you:

```python
import pandas as pd

df = pd.read_csv('SLS22.csv')
print(df.describe())
```

Additionally, with the data, you can use Python to compute various statistics, such as the mean score for run 1 using `df["run 1"].mean()`. Similarly, you can use masks to filter out specific rows based

Certainly! Here is a breakdown of the provided Swedish content:

---

## 2. Tasks

You are required to solve the following tasks and document your solutions in a PDF report. Alongside the report, you should also submit your code either as a Python script or as a Jupyter notebook with clear markdown explanations. You may collaborate on the project with one (and only one) other student from the course. However, you must write your own code and your own report. Both of you should not submit duplicate reports! Your report and code should be written in your own words to show that you understand the solutions. Also, mention the name of the person you collaborated with on the first page of your report.

1. **Warm-up**: The following tasks will familiarize you with the dataset and help you prepare the data for your predictive models.

    (a) All grades in the data frame currently range from 0 to 10. Normalize these values in the data frame to be between 0 and 1.

    (b) Plot a histogram for all the trick grades for tricks 1-4. What do you observe? Is there a certain value that appears more frequently than the others? If so, how does this value compare to the others?

    (c) For each trick from 1-4, create a new column named 'make i' for \(i=1,2,3,4\). The value of 'make i' in a particular row should be 1 if the skateboarder landed trick \(i\) and 0 otherwise.

    (d) Estimate the probability for each skateboarder that a trick gets a grade greater than 0.6, given that the skateboarder lands the trick. What's the probability that the skateboarder doesn't land a particular trick? What are your observations? Relate your findings to your observations in part (b).

    (e) Plot a scatter diagram for run score 1 against run score 2. Can you discern any clear correlation from the chart?

2. **A Frequentist Model**: We aim to build a model that can predict which among the 16 skateboarders in the LCQ wins a spot in the final. One way is to construct a model for each skateboarder, simulate run scores and trick scores for each of them, and then combine these simulations to mimic the LCQ. Multiple LCQs can be simulated and the top four skateboarders with the highest total scores can be extracted from each. The mode of these results becomes our prediction. This model assumes that performances of skateboarders are independent. For simplicity, we assume that the score of a particular run \(Y_{i}\) and the score of a particular trick \(X_{i}\) are independent for each skateboarder \(i\). We also assume that all trick scores and run scores are independent, identically distributed outcomes from \(X_{i}\) and \(Y_{i}\), respectively. We can begin by defining a model for \(X_{i}\) and \(Y_{i}\).

    Based on observations in Task 1, a reasonable model for \(X_{i}\) is:

    \[
    X_{i}= 
    \begin{cases}
      0 & \text{if } V_{i}=0, \\
      Z_{i} & \text{if } V_{i}=1,
    \end{cases}
    \]

    where \(V_{i} \sim \text{Ber}(\theta_{i})\), \(Z_{i} \sim \text{Beta}(\alpha_{i}, \beta_{i})\), and \(V_{i}\) is independent of \(Z_{i}\). It can be shown that:

    \[
    f_{X_{i}}(x_{i} | \theta_{i}, \alpha_{i}, \beta_{i}) = (1-\theta_{i}) 1_{x_{i}=0} + \theta_{i} f_{Z_{i}}(z_{i}) .
    \]

    The choice \(V_{i} \sim \text{Ber}(\theta_{i})\) models that a skateboarder scores 0 only if they fail to land the trick. The choice \(Z_{i} \sim \text{Beta}(\alpha_{i}, \beta_{i})\) models that the score for a certain trick is the portion of the trick that was "perfect".

    (a) Provide a point estimate for each \(\theta_{i}\), the probability that skateboarder \(i\) lands a trick.

    (b) Provide a point estimate for the parameters \([\alpha_{i}, \beta_{i}]\) for each skateboarder \(i\). Are there skateboarders for whom your chosen point estimate doesn't exist? If so, suggest an alternative point estimate for these \(\theta_{i}\). Justify your choices for point estimates.

    (c) Propose a model for \(Y_{i}\) and provide a point estimate for the parameters of your model. Justify your choices for the model and point estimate.

    (d) Use your model for \([X_{i}, Y_{i}]\) to simulate 5000 LCQs. For each simulation, extract the four skateboarders \(W=\left[W_{1}, W_{2}, W_{3}, W_{4}\right]\) with the highest total scores. What's the mode for \(W_{1}, ... , W_{5000}\)? The real winners of the LCQ are... [Content seems to be cut off here.]

---

This is a translated and organized breakdown of the text you provided. The content seems to describe a project or assignment related to data analysis and modeling based on scores and performances of skateboarders.

Here's a simplified and brief English translation for the provided Swedish text:

## 3. Gustavo Hoban Eaton Decenzo.

- How many of the actual winners are predicted by the typical value? What's the estimated probability for the real winners based on your simulations? From the typical value?

3. A Bayesian Model:
   - Instead of the frequentist model developed in Task 2, we can consider a Bayesian model.

(a) Propose a simultaneous prior distribution for the parameters \(\left[\Theta_{i}, A_{i}, B_{i}\right]^{\mathrm{T}}\) for \(X_{i}\). Justify your choice.

(b) Generate 5,000 random outcomes from the posterior distribution:

\[

f_{\theta_{i}, \alpha_{i}, \beta_{i} | \boldsymbol{X}_{i}}

\]

Plot your results for the marginal posterior distributions:

\[

f_{\theta_{i} | \boldsymbol{X}_{i}} \quad \text{and} \quad f_{\alpha_{i}, \beta_{i} | \boldsymbol{X}_{i}}

\]

Calculate the posterior sample mean and variance for each parameter \(theta_{i}, \alpha_{i}\), and \(beta_{i}\) for all skateboarders.

(c) Propose a (simultaneous) prior distribution for the parameters of your model \(Y_{i}\) from task 2(c). Justify your choice. Generate 5,000 outcomes from the posterior distribution and create a scatter plot of the results. What's the sample mean and variance for each of your parameters based on your outcomes?

(d) Use your Bayesian model to simulate 5,000 LCQs. What's the mode of your outcomes? How many real winners are predicted? What's the estimated probability for the real winners based on your outcomes?

(e) In the model in task 3(d), we assumed certain parameters are independent given the data. Draw a directed acyclic graph with as few edges as possible for the simultaneous distribution of several variables. Based on your graph, can you conclude that the marginal posterior distribution factorizes as given? Consider your parameters for \(Y_{i}\) and \(X_{i}\). According to your graph, is our assumption valid? Can we assume the independence relation if only data \(o_{i}\) is given instead?

4. A Bayesian model with hierarchy:
   - To account for possible variations in skateboarder performances between different contests, we can construct a hierarchical model.

(a) Assume \(Theta_{i} | A_{i}=\alpha_{i}, B_{i}=\beta_{i} \sim Beta(\alpha_{i}, \beta_{i})\). Choose a suitable simultaneous prior distribution for \(\left[\Theta_{i}, A_{i}, B_{i}\right]^{T}\). Justify your choice.

(b) Generate 5,000 random outcomes from the simultaneous posterior distribution:

\[

f_{A_{i}, B_{i} | \boldsymbol{X}_{i}}

\]

Use your simulations to generate outcomes for the marginal posterior distribution \(Theta_{i} | \boldsymbol{X}_{i}=\boldsymbol{x}_{i}\). Plot your outcomes. Give estimates for the posterior mean and variance. How do these variances compare to the model in Task 3?

(c) Using your 5,000 outcomes from part (b), simulate 5,000 LCQ winners. What's the mode of the results? What are the respective estimated probabilities for the real winners?

5. Discussion:
   - Always important to reflect on our model assumptions.

(a) Compare the results of the different models. Which skateboarders are correctly predicted? Offer explanations for the differences between model predictions. Which model do you prefer?

(b) Compare your estimates for \(\theta_{i}\) in Task 1 with the expected values and variances in Task 3 and 4. Considering those who are predicted to win, do these statistics offer insights into successful strategies?

(c) Estimate the mean and standard deviation for each skateboarder's total score for models in Task 3 and 4. Does this statistic support your predictions?

(d) In all models, we assumed performances are independent. Is this a reasonable assumption?

(e) In all models, we ignored the order in which skateboarders took turns. Is this reasonable? Why or why not?

Please note that this is a broad overview and might not capture every nuance from the original text.