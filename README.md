# Fashion-MNIST Neural Network Experiments

This repository contains a Jupyter notebook implementing a sequence of neural-network experiments on the [Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

The work progresses from a NumPy implementation of backpropagation to PyTorch experiments covering activation functions, loss functions, optimizers, overfitting, regularization, and hyperparameter tuning.

## Repository Contents

- [`DLP_A1_23F_0696_23F_0608.ipynb`](DLP_A1_23F_0696_23F_0608.ipynb): Complete analysis, implementations, training runs, plots, and written results.
- `README.md`: Reproduction and project documentation.

## Dataset

**Dataset:** Fashion-MNIST by Zalando Research  
**Source:** [Kaggle - Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

The notebook expects the following two CSV files:

```text
fashion-mnist_train.csv
fashion-mnist_test.csv
```

Each file contains a `label` column and 784 pixel-value columns representing a 28 x 28 grayscale image. Pixel values are normalized from the range 0-255 to the range 0-1 before training.

## Experiments Included

### Part 1: Backpropagation from Scratch

- Builds a 784-64-10 multilayer perceptron using NumPy.
- Implements ReLU, softmax, categorical cross-entropy, backpropagation, and manual gradient descent.
- Trains on a 5,000-sample subset for 20 epochs.
- Verifies the manually calculated gradients against PyTorch autograd.

### Part 2: Activation Function Comparison

Uses a common 784-128-64-10 PyTorch MLP to compare:

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU

The notebook also measures first-layer gradient magnitudes and inactive ReLU units.

### Part 3: Loss Functions

Compares the same classifier trained with:

- Cross-entropy loss
- Mean squared error applied to softmax probabilities and one-hot labels

A separate MLP is trained for California Housing regression using MSE, RMSE, and MAE evaluation metrics.

### Part 4: Optimizer Comparison

Compares the following optimizers:

- SGD
- SGD with momentum
- RMSProp
- Adam

The experiments first use a common learning rate and then use selected learning rates for each optimizer.

### Part 5: Forced Overfitting

- Restricts training to 2,000 samples.
- Uses a large 4-layer MLP with 512 hidden units per layer.
- Tracks training and validation loss and accuracy.
- Reports the generalization gap and identifies when the loss curves separate.

### Part 6: Regularization Study

Evaluates methods intended to reduce overfitting:

- L2 weight decay with multiple strengths
- L1 penalty
- Dropout with rates 0.2, 0.4, and 0.6
- Batch normalization
- Early stopping with patience 5
- Data augmentation using horizontal flips and small rotations
- Increasing the training set to 10,000 and 20,000 samples

### Part 7: Hyperparameter Tuning

Runs a random search over 12 configurations using 5-fold cross-validation.

Search space:

| Hyperparameter | Values |
|---|---|
| Learning rate | 0.0001, 0.0005, 0.001, 0.005 |
| Hidden width | 256, 512, 768 |
| Dropout rate | 0.2, 0.4, 0.6 |

The selected configuration is retrained on the full 48,000-sample training split and evaluated once on the held-out test set using accuracy, macro precision, macro recall, macro F1, and a confusion matrix.

## Reproducibility

### Option A: Run on Kaggle

The notebook was written using Kaggle's notebook environment and is easiest to reproduce there.

1. Open the [Fashion-MNIST Kaggle dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
2. Create or open a Kaggle notebook.
3. Add the dataset to the notebook using **Add Input**.
4. Upload or copy `dlp-a-1-23f-0696-23f-0608.ipynb` into the Kaggle notebook, or copy its cells into a new notebook.
5. Confirm that the dataset is available at:

   ```text
   /kaggle/input/datasets/zalando-research/fashionmnist
   ```

6. Run the notebook from the first cell to the last cell using **Run All**.
7. Review the printed metrics, plots, comparison tables, and final confusion matrix.

If Kaggle mounts the dataset at a different location, update both `DATASET_PATH` and `DATA_PATH` in the environment setup cells.

### Option B: Run Locally

#### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

#### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn torch torchvision kagglehub jupyter notebook
```

A CUDA-enabled PyTorch installation may be used instead if a compatible NVIDIA GPU and CUDA setup are available. The notebook automatically selects CUDA when available and otherwise uses the CPU.

#### 4. Download the dataset

Download the CSV files from the [Kaggle dataset page](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and place them in a local directory, for example:

```text
data/fashionmnist/
├── fashion-mnist_train.csv
└── fashion-mnist_test.csv
```

#### 5. Update the dataset path

In the notebook's environment setup section, replace the Kaggle path in both variables:

```python
DATASET_PATH = "data/fashionmnist"
DATA_PATH = "data/fashionmnist"
```

The path may also be absolute, for example:

```python
DATA_PATH = r"D:\datasets\fashionmnist"
```

#### 6. Open and run the notebook

```bash
jupyter notebook dlp-a-1-23f-0696-23f-0608.ipynb
```

Select the virtual environment as the notebook kernel and run all cells in order. The notebook contains stateful cells, so running cells out of order can result in missing variables or overwritten model definitions.

## Execution Notes

- Run the cells sequentially from top to bottom.
- The complete notebook performs many training runs, including 100-epoch regularization experiments and 5-fold cross-validation. CPU execution may take a significant amount of time.
- A GPU is recommended for the full notebook, but it is not required.
- Training uses random seeds in several experiments, but exact results can still vary with the PyTorch version, hardware, data-loader behavior, and numerical precision.
- The notebook currently contains reported result tables in its markdown cells. Rerunning the code regenerates the numerical outputs and plots.
- Do not commit the downloaded CSV files to Git unless repository size and dataset licensing requirements have been reviewed.

## Reported Results

The result summaries recorded in the notebook include the following headline findings:

- The NumPy backpropagation gradients closely match PyTorch autograd.
- ReLU achieved the strongest validation accuracy among the activation functions tested in Part 2, at approximately 88.19%.
- Cross-entropy was better suited to Fashion-MNIST classification than MSE in the reported comparison.
- Adam was the strongest optimizer in the reported tuned optimizer comparison.
- The forced-overfitting experiment produced a substantial training-validation generalization gap.
- Increasing the training data to 20,000 samples provided one of the best reported trade-offs between reducing the gap and preserving training accuracy.
- The Part 7 search selected learning rate `0.0005`, hidden width `256`, and dropout `0.2`.
- The reported final Part 7 test accuracy was approximately 88.36%, with macro F1 of approximately 88.44%.

These values are the notebook's recorded results and should be treated as reference values. The exact values from a fresh run may differ slightly.

## Class Labels

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Technologies

- Python
- NumPy
- pandas
- scikit-learn
- PyTorch
- torchvision
- Matplotlib
- Jupyter Notebook

## Citation and Attribution

The dataset used in this project is provided by Zalando Research through Kaggle:

> Fashion-MNIST: a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples.

Dataset link: [https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
