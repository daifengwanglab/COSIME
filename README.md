# COSIME: Cooperative multi-view integration and Scalable and Interpretable Model Explainer

Cooperative Multiview Integration and Scalable and Interpretable Model Explainer (COSIME) is a machine learning model that integrates multi-view data for disease phenotype prediction and computes feature importance and interaction scores. By leveraging deep learning-based encoders, COSIME effectively captures the complex, multi-layered interactions between different omic modalities while preserving the unique characteristics of each data type. The integration of LOT techniques aligns and merges heterogeneous datasets, improving the accuracy of modeling across-view relationships in the joint latent space. In addition, COSIME leverages the Shapley-Taylor Interaction Index to compute feature importance and interaction values, allowing for a deeper understanding of how individual features and their interactions contribute to the predictions.

![Title](Images/Fig1_Coop_Git.png "Title")

## Installation
1. Clone and navigate to the respository.
```bash
git clone https://github.com/jeromejchoi/COSIME.git
cd COSIME
```
2. Create and activate a virtual environment for python 3.10.14 with `conda` or `virtualenv`.
```bash
# conda
conda create -n COSIME python=3.10.14
conda activate COSIME

# virtualenv
source COSIME/bin/activate
COSIME\Scripts\activate
```
3. Install dependencies for production and development with `pip`.
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
#### Estimated time of installing:
- About 2~3 minutes for requirements and requirements-dev, respectively on an Apple M1 Max with 10 physical cores and 32 GB of Unified Memory.

## Example: Simulated data (Binary outcome - high signal & late fusion)
### Training and Predicting
```bash
python Code/Model/main.py \
  --input_data_1="Example/binary_high_late_x1.csv" \
  --input_data_2="Example/binary_high_late_x2.csv" \
  --type="binary" \
  --predictor="regression" \
  --fusion="early" \
  --batch_size=32 \
  --epochs=100 \
  --learning_rate=0.0001 \
  --learning_gamma=0.99 \
  --dropout=0.5 \
  --kld_1_weight=0.02 \
  --kld_2_weight=0.02 \
  --ot_weight=0.02 \
  --cl_weight=0.9 \
  --dim=100 \
  --earlystop_patience=40 \
  --delta=0.001 \
  --decay=0.001 \
  --splits=5 \
  --save="Example" \
  --log="Example/training.log" \
  --device="cpu"
```
#### Parameters Overview

##### Input Data
- **input_data_1**: Input data view 1.
- **input_data_2**: Input data view 2.

##### Model Configuration
- **type**: Outcome type, either binary (classification) or continuous (regression).
- **predictor**: Type of model used for continuous outcomes (regression or NN).
- **fusion**: Fusion method (early or late).

##### Training Settings
- **batch_size**: Number of samples processed in one pass.
- **epoch**: Number of training epochs.
- **learning rate**: Controls how much weights are adjusted during training.
- **learning gamma**: Rate at which the learning rate decays during training.
- **dropout**: Probability of randomly dropping neurons during training to prevent overfitting.

##### Loss Weights
- **kld_1_weight**: Weight for the KLD loss (view A).
- **kld_2_weight**: Weight for the KLD loss (view B).
- **ot_weight**: Weight for the LOT loss.
- **cl_weight**: Weight for the prediction loss.

##### Latent Space
- **dim**: Size of the joint latent space where multiple views are represented.

##### Stopping and Regularization
- **earlystop_patience**: Number of epochs to wait without improvement before stopping training.
- **delta**: Minimum improvement required to reset early stopping counter.
- **decay**: How much the learning rate decreases during training.
- **splits**: Number of splits for cross-validation.

##### File Paths and Device.
- **save**: Path to save model outputs and history (training history, holdout evaluation history, and the best model).
- **log**: Path to save the training logs.
- **device**: Device to run the model.


#### Results
<p style="text-align: left;">
  <img src="Images/box_binary_high_late.png" alt="Title 1" width="45%" />
</p>
<p style="text-align: left;">
  Holdout evaluation (5-fold CV)
</p>

#### Estimated time of running:
- About 1.5 hours on an Apple M1 Max with 10 physical cores and 32 GB of Unified Memory.
- About 0.9 hours on an Intel Xeon Gold 6140 system with 36 physical cores, 200 GB of RAM, and 25.3 MB of L3 cache.


### Computing Feature Importance and Interaction
```bash
python Code/Explainer/main.py \
--input_data="Example/binary_high_late.df.csv" \
--input_model="Example/best_model_binary_high_late.pt" \
--model_script_path="Example/model_binary_high_late.py" \
--input_dims="100,100" \
--fusion="late" \
--dim 150 \
--dropout 0.5 \
--mc_iterations 50 \
--batch_size 32 \
--max_memory_usage_gb 2 \
--interaction True \
--save="Example" \
--log="Example/binary_high_late.log"
```
#### Parameters Overview

##### Input Data and Model
- **input_data**: Holdout multi-view dataset (without labels).
- **input_model**: Trained model.
- **model_script_path**: Model class used in training the model.
- **input_dims**: Dimensions in two input data views.

##### Model Configuration
- **fusion**: Fusion method (early or late).
- **dim**: Size of the joint latent space where multiple views are represented.
- **dropout**: Probability of randomly dropping neurons during training to prevent overfitting.

##### Monte Carlo Sampling and Memory
- **mc_iterations**: Number of Monte Carlo sampling iterations.
- **batch_size**: Number of samples processed together in one forward pass through the model.
- **max_memory_usage_gb**: Maximum memory usage in gigabytes (GB) for the model during computation
- **interaction**: Compute both feature importance and pairwise feature interaction (True) or just feature importance (False).

##### File Paths
- **save**: Path to save the outputs.
- **log**: Path to save the training logs.

  
#### Results
| ![Title 1](Images/FI_binary_high_A.png "Title 1") | ![Title 2](Images/FI_binary_high_B.png "Title 2") |
|:-------------------------------------------------:|:-------------------------------------------------:|
| Top 20 absoulte feature importance values (View A) | Top 20 absoulte feature importance values (View B) |

| ![Title 3](Images/SI_binary_high_A.png "Title 3") | ![Title 4](Images/SI_binary_high_B.png "Title 4") |
|:-------------------------------------------------:|:-------------------------------------------------:|
| Pairwise feature interactions for the first 50 features (View A) | Pairwise feature interactions for the first 50 features (View B) |

#### Estimated time of running:
- About 7.53 hours on an Apple M1 Max with 10 physical cores and 32 GB of Unified Memory.
- About 4.44 hours on an Intel Xeon Gold 6140 system with 36 physical cores, 200 GB of RAM, and 25.3 MB of L3 cache.


## References
Dhamdhere, K., Agarwal, A. & Sundararajan, M. The Shapley Taylor Interaction Index Ver- sion Number: 2. (2019).
