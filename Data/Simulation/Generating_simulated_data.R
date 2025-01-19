library(forecast)
library(moments)
library(pROC)
library(PRROC)

# n: # of samples
# f_x1: # of features in x_1
# f_x2: # of features in x_2
# f_latent: # of latent factors
# scale_1: scaling Factor for x_1 (strength of the influence of the latent factors on x_1)
# scale_2: scaling Factor for x_1 (strength of the influence of the latent factors on x_2)
# std: standard deviation of latent factors for x_1 and x_2 (correlated)
# std_1: standard deviation of latent factors for x_1 (uncorrelated)
# std_2: standard deviation of latent factors for x_2 (uncorrelated)
# latent_strength: Strength of the latent factors' impact on y (unobserved or hidden variables that influence the observed data)
# noise: standard deviation of noise added to y
# correlation: x_1 and x_2 are correlation (TRUE) or uncorrelated (FALSE)
# data_type: continuous or binary
# interaction: TRUE / FALSE
# interaction_weight: weight for the interaction effects

generate_data <- function(n = 1000, f_x1 = 100, f_x2 = 100, f_latent = 20,  
                          scale_1 = 5, scale_2 = 5,  
                          std = 1, std_1 = 1, std_2 = 1,
                          latent_strength = 3,
                          noise = 5, correlation = TRUE, data_type = "binary",  
                          interaction = TRUE, interaction_weight = 10,  
                          seed = NULL, U) {
  
  # Set seed for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Initialize latent factor variables
  U <- NULL
  U1 <- NULL
  U2 <- NULL
  beta <- NULL
  beta_U_combined <- NULL
  
  # Generate latent factors based on correlation setting
  if (correlation) {
    # When correlated, use a single matrix U for both x_1 and x_2
    if (interaction) {
      U = matrix(rnorm(n * (f_latent + 5), sd = std), n, f_latent + 5)
      beta = rep(latent_strength, f_latent + 5)
    } else {
      U = matrix(rnorm(n * f_latent, sd = std), n, f_latent)
      beta = rep(latent_strength, f_latent) 
    }
  } else {
    # When uncorrelated, use separate U1 and U2 for x_1 and x_2
    if (interaction) {
      U1 = matrix(rnorm(n * (f_latent + 5), sd = std_1), n, f_latent + 5) 
      U2 = matrix(rnorm(n * (f_latent + 5), sd = std_2), n, f_latent + 5)
      beta_U_combined = rep(latent_strength, f_latent + 5)
    } else {
      U1 = matrix(rnorm(n * f_latent, sd = std_1), n, f_latent)
      U2 = matrix(rnorm(n * f_latent, sd = std_2), n, f_latent) 
      beta_U_combined = rep(latent_strength, f_latent)
    }
  }
  
  # Generate x_1 and x_2 as independent standard normal variables
  x_1 = matrix(rnorm(n * f_x1), n, f_x1)
  x_2 = matrix(rnorm(n * f_x2), n, f_x2)
  
  # Add latent factors to x_1 and x_2
  if (correlation) {
    # When correlation, add the same latent factor U to both x_1 and x_2
    if (interaction) {
      for (m in seq(f_latent + 5)) { 
        u = U[, m]
        x_1[, m] = x_1[, m] + scale_1 * u
        x_2[, m] = x_2[, m] + scale_2 * u
      }
    } else {
      for (m in seq(f_latent)) {
        u = U[, m]
        x_1[, m] = x_1[, m] + scale_1 * u
        x_2[, m] = x_2[, m] + scale_2 * u
      }
    }
    
    # Compute interaction terms only for the correlation case
    if (interaction) {
      for (i in (f_latent + 1):(f_latent + 5)) {
        for (j in (f_latent + 1):(f_latent + 5)) {
          # Introduce the interaction term to both x_1 and x_2
          x_1[, i] = x_1[, i] + interaction_weight * U[, i] * U[, j]
          x_2[, j] = x_2[, j] + interaction_weight * U[, i] * U[, j]
        }
      }
    }
    
  } else {
    # When uncorrelated, add separate latent factors U1 and U2 to x_1 and x_2
    if (interaction) {
      for (m in seq(f_latent + 5)) { 
        u1 = U1[, m]
        u2 = U2[, m]
        x_1[, m] = x_1[, m] + scale_1 * u1
        x_2[, m] = x_2[, m] + scale_2 * u2
      }
    } else {
      for (m in seq(f_latent)) {
        u1 = U1[, m]
        u2 = U2[, m]
        x_1[, m] = x_1[, m] + scale_1 * u1
        x_2[, m] = x_2[, m] + scale_2 * u2
      }
    }
  }
  
  colnames(x_1) <- paste0("feature_", 1:f_x1)
  colnames(x_2) <- paste0("feature_", 1:f_x2)
  
  # # Compute y
  # if (correlation) {
  #   mu_all = U %*% beta
  # } else {
  #   mu_all = U1 %*% beta_U_combined + U2 %*% beta_U_combined
  # }

  # Compute y
if (correlation) {
  # If correlation is true, use shared U for both x_1 and x_2
  if (interaction) {
    # Include interaction terms in the computation for correlated case
    mu_all = U %*% beta + rowSums(U[, (f_latent + 1):(f_latent + 5)] * U[, (f_latent + 1):(f_latent + 5)])
  } else {
    # No interaction, simply use the latent factors U
    mu_all = U %*% beta
  }
} else {
  # If correlation is false, use separate U1 and U2 for x_1 and x_2
  if (interaction) {
    # Include interaction terms in the uncorrelated case
    mu_all = U1 %*% beta_U_combined + U2 %*% beta_U_combined +
      rowSums(U1[, (f_latent + 1):(f_latent + 5)] * U2[, (f_latent + 1):(f_latent + 5)])
  } else {
    # No interaction, use separate U1 and U2
    mu_all = U1 %*% beta_U_combined + U2 %*% beta_U_combined
  }
}
  
  e = noise * rnorm(n)
  y = mu_all + e
  
  # Include interaction terms
  mu_noint = mu_all
  
  # Transform y into binary outcomes
  if(data_type == "binary") {
    probabilities = 1 / (1 + exp(-y))
    
    y = rbinom(n, size = 1, prob = probabilities)
    
    # Fit logistic models
    model_mu_all = glm(y ~ mu_all, family = binomial)
    model_e = glm(y ~ e, family = binomial)
    
    # Predict probabilities
    pred_mu_all = predict(model_mu_all, type = "response")
    pred_e = predict(model_e, type = "response")
    
    # Compute AUROC
    auc_mu_all = pROC::auc(y, pred_mu_all)
    auc_e = pROC::auc(y, pred_e)
    auc_mu_noint = pROC::auc(y, mu_noint)
    
    # Compute AUPRC
    prc_mu_all = PRROC::pr.curve(scores.class0 = pred_mu_all, weights.class0 = y)$auc.integral
    prc_e = PRROC::pr.curve(scores.class0 = pred_e, weights.class0 = y)$auc.integral
    prc_mu_noint = PRROC::pr.curve(scores.class0 = mu_noint, weights.class0 = y)$auc.integral
    
    # Calculate AUROC-based SNR
    snr_auroc = (auc_mu_all - 0.5) / (auc_e - 0.5)
    
    # Calculate AUPRC-based SNR
    snr_prc = (prc_mu_all) / (prc_e)
    
    # Print results
    cat("Label ratio:", table(y), "\n")
    cat("AUROC for mu_all:", auc_mu_all, "\n")
    cat("AUROC for e:", auc_e, "\n")
    cat("AUROC for mu_noint:", auc_mu_noint, "\n")
    cat("AUPRC for mu_all:", prc_mu_all, "\n")
    cat("AUPRC for e:", prc_e, "\n")
    cat("AUPRC for mu_noint:", prc_mu_noint, "\n")
    cat("SNR (AUROC):", snr_auroc, "\n")
    cat("SNR (AUPRC):", snr_prc, "\n")
  } else if(data_type == "continuous") {
    residuals = y - mu_all
    var_signal = var(mu_all)
    var_noise = var(residuals)
    
    # Compute SNR
    snr_continuous = var_signal / var_noise
    cat("SNR (continuous):", snr_continuous, "\n")
  }
  
  # Return the generated data
  return(list(x_1_df = cbind.data.frame(x_1, y), x_2_df = cbind.data.frame(x_2, y), U=U))
}
