# COSIME: Cooperative multi-view integration and Scalable and Interpretable Model Explainer

Cooperative Multiview Integration and Scalable and Interpretable Model Explainer (COSIME) is a machine learning model that integrates multi-view data for disease phenotype prediction and computes feature importance and interaction scores. By leveraging deep learning-based encoders, COSIME effectively captures the complex, multi-layered interactions between different omic modalities while preserving the unique characteristics of each data type. The integration of LOT techniques aligns and merges heterogeneous datasets, improving the accuracy of modeling cross-modality relationships in the joint latent space. After training a model, COSIME leverages the Shapley-Taylor Interaction Index to compute feature importance and interaction values, allowing for a deeper understanding of how individual features and their interactions contribute to the model's predictions.

![Title](Images/Fig1_Coop_Git.png "Title")

## Installation
1. Clone and navigate to the respository.
```bash
https://github.com/jeromejchoi/COSIME
cd COSIME
```
2. Create and activate a virtual environment for python 3.10.14 with `conda` or `virtualenv`.
```bash
# conda
conda create -n COSIME python=3.10.14
conda activate COSIME

# virtualenv
source COSIME/bin/activate  # on Linux/Mac
COSIME\Scripts\activate  # on Windows
```
3. Install dependencies for production and development with `pip`.
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
## Example: Simulated data (Binary outcome - high signal & late fusion)
### Training and Predicting

### Computing Feature Importane and Interaction
```bash

python main.py \
  --input_data="/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/ROSMAP_late2/Results/padded_data_no_na.csv" \
  --input_model="/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/ROSMAP_late2/Results/best_overall_config_lr_0.000999421929570799_KLD_A_0.029142176782704064_KLD_B_0.02858008241164476_OT_0.05046950040744483_CL_0.9999507032635159.pt" \
  --model_script_path="/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Manuscript/Refactoring/FI//user_model.py" \
  --input_dims="305,305" \
  --fusion="late" \
  --save="/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/ROSMAP_late2/Results" \
  --log="/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/ROSMAP_late2/Results/logfile.log" \
  --dim 150 \
  --dropout 0.5 \
  --mc_iterations 10 \
  --batch_size 32 \
  --interaction True
```
