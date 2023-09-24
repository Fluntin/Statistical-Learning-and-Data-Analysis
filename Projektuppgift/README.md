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
