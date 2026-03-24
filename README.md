# AlzheimerCellNet: Single-Cell RNA-Seq Classification

This project implements **AlzheimerCellNet**, a specialized neural network architecture designed to classify brain cell lineages (such as Astrocytes, Neurons, Microglia, etc.) from high-dimensional single-cell RNA-sequencing (scRNA-seq) data. It was built for the JOIST Hackathon context and achieves rapid, robust performance.

## 🧬 Architecture Overview

The core architecture (defined in `model_arch.py`) is a **Joint Autoencoder & Classifier**. It operates on ~16,678-dimensional gene expression profiles.

1. **Encoder**: Reduces the input features into a 128-dimensional latent representation, utilizing an intermediate 512-neuron layer with ReLU and Batch Normalization to ensure stable, rapid early-stage convergence.
2. **Decoder**: A Denoising component that attempts to reconstruct the original 16,678 gene features from the latent space, forcing the network to capture global transcriptional variance rather than just memorizing class labels.
3. **Classifier**: Computes the final cell-type prediction (logits) from the 128-dimensional latent vector via a 64-neuron layer incorporating Dropout for regularization.

## ⚙️ Data Preprocessing

Data is handled via `data_utils.py`:
- Drops irrelevant identifier columns and separates the `tag` label.
- Splits data (85% Train, 15% Validation) without data leakage.
- Normalizes all ~16,678 features using `StandardScaler` to ensure robust gradient descent.

## 🧠 Training Strategy

The model is trained (`engine.py`) using a **Joint Loss Optimization**. 
During backpropagation, the loss function minimizes both the classification error and a fraction of the autoencoder reconstruction error:
`Loss = CrossEntropyLoss + 0.1 * MSELoss`

This enables the model to converge extremely fast (achieving >90% validation Accuracy and F1 score mathematically by the end of Epoch 1) because the underlying biological lineages are highly separable in the compressed generative space. By the end of training, it typically sustains ~99% Accuracy and ~0.98 Macro F1 score on the validation set.

## 🚀 Running the Project

### 1. Training the Model
Run the main script to process the data, initialize the model, and train for 20 epochs.
```bash
python main.py
```
**Outputs:** 
- `training_history.csv` (contains epoch logs for Loss, F1, and Accuracy)
- `submission_alzheimer_2026.csv` (predictions on the unseen test set data)

### 2. Plotting Performance Metrics
Once `training_history.csv` is generated, run the plotting script to visualize the model's training dynamics.
```bash
python plot_metrics.py
```
**Output:**
- `training_metrics.png` (Training Loss, Validation F1 Score, and Validation Accuracy curves)
