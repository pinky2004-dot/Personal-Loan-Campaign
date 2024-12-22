# Personal Loan Campaign

## Objective

Predict whether a liability customer will buy personal loans, to understand which customer attributes are most significant in driving purchases, and identify which segment of customers to target more.

## Topics
- Pandas
- Numpy
- Matplotlib
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Inferential Statistics
- Correlation and Relationships
- Model Building (Decision Tree, Pre-Pruning, Post-Pruning)

## Which model did I use and what techniques to enhance it's performance?

For my project, I initially used a default Decision Tree model as the baseline model without passing any hyperparameters. This allowed me to evaluate its performance on both the training and testing datasets to establish a benchmark for comparison. The default model served as a foundation to identify areas where optimization and fine-tuning could enhance performance.

To improve the model's performance, I implemented pre-pruning techniques in the second Decision Tree model. Pre-pruning involved restricting the growth of the tree during training by setting constraints on hyperparameters such as the maximum depth, minimum number of samples required to split a node, and the minimum samples per leaf. These constraints effectively reduced the complexity of the tree and prevented overfitting. As a result, Model 2 demonstrated strong generalization capabilities, with a **recall of 84.3% on the training dataset and 85.4% on the testing dataset**. This indicates that Model 2 generalized well to unseen data without overfitting, making it the best-performing model of the three.

Next, I applied post-pruning techniques in the third iteration of the Decision Tree model. For post-pruning, I allowed the tree to grow fully during training to capture all potential splits. Afterward, I used a pruning process based on the cost complexity parameter (ccp_alpha). Specifically:

1. I extracted a sequence of ccp_alpha values and their corresponding subtrees (classifiers, clfs) using the cost_complexity_pruning_path function.
2. For each alpha, I evaluated the trade-off between the complexity of the tree and its performance. I plotted:
    - Number of nodes vs. alpha: This showed how the size of the tree (number of nodes) decreased as alpha increased, indicating the pruning process.
    - Depth of the tree vs. alpha: This illustrated how the tree's depth decreased with higher alpha values, reducing overfitting by eliminating insignificant splits.

By analyzing these relationships, I selected an optimal alpha that balanced tree complexity and performance. While post-pruning helped reduce overfitting, it did not outperform the pre-pruned model (Model 2) in terms of recall.

In conclusion, the pre-pruned model (Model 2) emerged as the best model due to its ability to balance recall on both training and testing datasets, effectively avoiding overfitting and generalizing well to unseen data. This systematic exploration of pre-pruning and post-pruning techniques highlights the importance of balancing tree complexity and predictive performance.
