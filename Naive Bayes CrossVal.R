# Crossvalidating a Naive Bayes model 

library(klaR)
library(tidymodels)
library(yardstick)
library(dplyr)
library(knitr)


test <- list(); train <- list(); pred_cv <- list(); results <- list(); metrics <- list()
kappa <- c(); Acc <- c(); AUC <- c()

for(i in 1:CV){
  test_rows <- seq[i]:seq[i+1]
  test[[i]] <- titanic_train2_training[test_rows,]
  train[[i]] <- titanic_train2_training[-test_rows,]
  
  model_cv <- naive_Bayes(Laplace = 7) %>%
    set_engine("klaR") %>%
    set_mode("classification") %>%
    fit(survived ~ ., data = train[[i]])
  
  pred_cv[[i]] <- model_cv %>%
    predict(test[[i]]) %>%
    bind_cols(test[[i]]) 
  
  results[[i]] <- bind_cols(truth = test[[i]]$survived, 
                            estimate=(as.numeric(pred_cv[[i]]$.pred_class)-2)^2, 
                            f_estimate = pred_cv[[i]]$.pred_class)
  
  metrics[[i]] <- results[[i]] %>% 
    metrics(truth = truth, estimate = f_estimate)
  kappa[i] <- metrics[[i]]$.estimate[2] 
  Acc[i] <- metrics[[i]]$.estimate[1]
  
  auc <- results[[i]] %>%
    roc_auc(truth = truth, estimate)
  AUC[i] <- auc$.estimate
}
df <- data.frame(rbind(Accuracy = mean(Acc), AUC = mean(AUC), kappa = mean(kappa)))
kable(df, col.names = c("Estimate"), 
      caption = "Cross-validated metrics for the optimized Naive Bayes model")
