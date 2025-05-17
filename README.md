# KuaiRec Short Video Recommender System

## Overview

This project develops a recommender system for short videos using the KuaiRec 2.0 dataset. The goal is to suggest personalized and relevant videos to users based on their preferences, interaction histories, and video content, similar to platforms like TikTok or Kuaishou.

---

## Methodology

### 1. Data Loading and Cleaning

- **Datasets Used:**
  - `big_matrix.csv`: Full user-item interaction data (sparse, diverse).
  - `small_matrix.csv`: Subset with highly active users (dense, for evaluation).
  - `item_categories.csv`: Video category information.
  - `item_daily_features.csv`: Daily video statistics (likes, plays).

- **Cleaning Steps:**
  - Remove duplicates and nulls.
  - Filter out invalid or out-of-range values (e.g., negative watch ratios).
  - Ensure all IDs are present and valid.

---

### 2. Feature Engineering

- **Category Binarization:**  
  Used `MultiLabelBinarizer` to convert list-based category features into binary columns for each category.

- **Combined Engagement Metric:**  
  - Calculated `like_ratio` for each video:  
    like_ratio = total_likes / total_plays
  - Created a `combined_ratio` for each user-video interaction:  
    combined_ratio = 0.8 * watch_ratio + 0.2 * like_ratio
  - This metric balances user engagement (watch time) and video quality (likes).

---

### 3. Model Training

- **Algorithm:**  
  - Used PySpark's Alternating Least Squares (ALS) for collaborative filtering.
  - Trained on the `big_matrix` with the `combined_ratio` as the rating.



- **Parameters:**  
  - `maxIter=10`, `regParam=0.13`, `rank=30`, `coldStartStrategy="drop"`, `nonnegative=True`.

---

### 4. Evaluation

- **Test Set:**  
  - Used `small_matrix` (active users) for unbiased evaluation.

- **Metrics:**
  - **RMSE (Root Mean Squared Error):** 1.0680
  - **MAE (Mean Absolute Error):** 0.2988
  - **Precision:**: 
    - Precision@10: 0.7411150961851972                                                                        
    - Precision@50: 0.736224323443104                                                                         
    - Precision@100: 0.7293772416041735
    - Precision@200: 0.7163351809585915                                                                           
    - Precision@500: 0.677535050537985
    - Precision@1000: 0.6113465927616564

---

## Experiments

- **Exploratory Data Analysis:**  
  - Analyzed user and item activity distributions, temporal patterns, and sparsity.

- **Feature Engineering:**  
  - Experimented with different weights for `watch_ratio` and `like_ratio`.
  - Used category binarization for potential future content-based enhancements (finally not been used).

- **Model Tuning:**  
  - Adjusted ALS hyperparameters for best RMSE/MAE trade-off.
  - Tried using other model training algorithms, however due to hardware limitations, could not.
- **

---

## Results

- **RMSE = 1.0666:**  
  Indicates the average prediction error is just over 1 unit. This is moderate, given the scale of the ratings.

- **MAE = 0.2970:**  
  On average, predictions are off by about 0.3 units.

- **Precision@10 = 0.74:**  
  About 7 out of 10 recommended videos in the top 10 are relevant to the user.

The RMSE and MAE could be lower if fine tuning the ALS.

---

## Conclusions

- **Strengths:**
  - The system provides relevant recommendations for active users, with a high Precision.
  - The methodology balances engagement and quality, improving the relevance of recommendations.

- **Limitations:**
  - RMSE and MAE indicate moderate prediction accuracy.
  - Further improvements could be made by:
    - Tuning ALS parameters further.
    - Exploring hybrid or deep learning approaches.
  - In the final project, the item_categories data set has not been used which should have been used to create better recommendations.
  - The temporal patterns has not been relevant in the final project.
  - There is only Precision@ as a evaluation metric, there should be more

---


## Authors

- Project by Max NAGAISHI for the Recommender Systems Course.
