# Fashion-MNIST Neural Network Experiments

This project contains a complete set of neural network experiments using the **Fashion-MNIST dataset**.

The assignment starts with implementing a neural network and backpropagation from scratch using NumPy. It then uses PyTorch to study activation functions, loss functions, optimizers, overfitting, regularization, and hyperparameter tuning.

## Repository Contents

- `DLP_A1_23F_0696_23F_0608.ipynb` — Complete notebook containing the code, experiments, results, plots, and explanations.
- `README.md` — Project information and instructions.

## Dataset

The project uses the **Fashion-MNIST dataset** from Zalando Research.

The dataset contains:
- 60,000 training images
- 10,000 test images
- 10 different clothing classes
- Each image is 28 × 28 pixels

The pixel values are normalized from **0–255 to 0–1** before training.

## Experiments

### Part 1 — Backpropagation from Scratch

A neural network with the architecture:

**784 → 64 → 10**

was implemented using NumPy without using an automatic differentiation framework.

The implementation includes:
- ReLU activation
- Softmax
- Cross-entropy loss
- Forward propagation
- Backpropagation
- Manual gradient descent

The manually calculated gradients were also compared with PyTorch autograd to verify the implementation.

### Part 2 — Activation Functions

A common neural network architecture was used to compare:

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU

Their validation performance and gradient behavior were studied.

### Part 3 — Loss Functions

Two loss functions were compared for classification:

- Cross-Entropy Loss
- Mean Squared Error (MSE)

A separate neural network was also trained on the **California Housing dataset** for regression using MSE, RMSE, and MAE.

### Part 4 — Optimizers

Four optimizers were compared:

- SGD
- SGD with Momentum
- RMSProp
- Adam

Both common and tuned learning rates were tested.

**Adam performed best in our experiment**, reaching 85% validation accuracy the fastest and achieving a final validation accuracy of **88.12%**.

### Part 5 — Forced Overfitting

The model was intentionally made more likely to overfit by:

- Using only 2,000 training samples
- Using a larger neural network
- Training for multiple epochs

Training and validation performance were monitored to observe the **generalization gap**.

### Part 6 — Regularization

Different techniques were tested to reduce overfitting:

- L2 regularization
- L1 regularization
- Dropout
- Batch normalization
- Early stopping
- Data augmentation
- Increasing the training dataset size

The results were compared to determine which methods helped improve generalization.

### Part 7 — Hyperparameter Tuning

A random search was performed to find a better combination of:

- Learning rate
- Hidden layer size
- Dropout rate

The selected configuration was then trained on the full training split and evaluated on the test set.

The selected configuration was:

- **Learning rate:** 0.0005
- **Hidden width:** 256
- **Dropout:** 0.2

The final reported test accuracy was approximately **88.36%**, with a macro F1-score of approximately **88.44%**.

## Main Results

Some important findings from the experiments were:

- The manually calculated backpropagation gradients closely matched PyTorch autograd.
- ReLU performed best among the activation functions tested in Part 2, with approximately **88.19% validation accuracy**.
- Cross-entropy was more suitable than MSE for the Fashion-MNIST classification task.
- Adam was the best optimizer in our comparison, achieving **88.12% validation accuracy**.
- The forced-overfitting experiment showed a clear difference between training and validation performance.
- Increasing the training data helped reduce overfitting and improved generalization.
- Hyperparameter tuning selected a learning rate of **0.0005**, hidden width of **256**, and dropout of **0.2**.
- The final tuned model achieved approximately **88.36% test accuracy**.

## How to Run

### Using Kaggle

The easiest way to run the notebook is through Kaggle.

1. Open the Fashion-MNIST dataset on Kaggle.
2. Create a Kaggle Notebook.
3. Add the Fashion-MNIST dataset using **Add Input**.
4. Upload the `.ipynb` file or copy the notebook cells.
5. Run the notebook cells in order.

The dataset should be available at:

```text
/kaggle/input/datasets/zalando-research/fashionmnist
