# CS5720-Assignment-1
# CS5720: Neural Network & Deep Learning
* **Institution**: University of Central Missouri
* **Department**: Department of Computer Science & Cybersecurity
* **Student Name**: [Navjot kaur]
* **Student ID**: [700774749]

---

## Executive Summary
This repository contains the full implementation and analysis for **CS5720 Home Assignment 1**. The assignment covers fundamental TensorFlow tensor manipulations, loss function comparison (MSE vs. Categorical Cross-Entropy), optimizer performance analysis (Adam vs. SGD on MNIST), and neural network training with live TensorBoard visualization.

---

## Task Breakdown & Explanations

### Task 1: Tensor Manipulations & Reshaping
* **Operations Executed**:
  * Generated a random 2D tensor of shape `(4, 6)`.
  * Verified tensor rank ($2$) and shape `(4, 6)` using TensorFlow functions (`tf.rank` and `tf.shape`).
  * Reshaped the tensor into `(2, 3, 4)` and transposed it to `(3, 2, 4)` using permutation axes (`perm=[5, 6]`).
  * Created a smaller tensor of shape `(1, 4)` and performed element-wise addition with the transposed `(3, 2, 4)` tensor via broadcasting.
 
    
* **Explanation of TensorFlow Broadcasting**:
  Broadcasting allows TensorFlow to perform element-wise arithmetic on tensors of different shapes. When operating on two tensors, TensorFlow compares their shapes element-wise from right to left. Dimensions are compatible if they are equal or if one of them is $1$. TensorFlow automatically expands dimensions of size $1$ without duplicating data in memory, enabling seamless tensor addition and multiplication.

---

### Task 2: Loss Functions & Hyperparameter Tuning
  * Defined ground truth one-hot targets (`y_true`) and probability predictions (`y_pred`).
  * Computed **Mean Squared Error (MSE)** and **Categorical Cross-Entropy (CCE)** loss values.
  * Modified predictions slightly to observe sensitivity and loss value updates.
  * Plotted loss comparisons using a Matplotlib bar chart.
    
  **Findings**:
  
   **Categorical Cross-Entropy (CCE)** penalizes confidently incorrect predictions exponentially harder than **MSE** due to its logarithmic loss formulation ($\text{CCE} = -\sum y_i \log(\hat{y}_i)$).
  * CCE is heavily preferred for classification tasks because its steep gradient yields faster weight updates when model predictions are far from target labels.

---

### Task 3: Train a Model with Different Optimizers (Adam vs. SGD)
* **Dataset**: MNIST Handwritten Digits ($28 \times 28$ grayscale images normalized to $[5]$).
* **Model Setup**: Built two identical feedforward neural network models.
  * Model 1 compiled with the **Adam** optimizer (`optimizer='adam'`).
  * Model 2 compiled with standard **SGD** (`optimizer='sgd'`).
 
    
* **Findings & Accuracy Trends**:
  * **Adam** converged significantly faster than SGD, achieving high training and validation accuracy within the first 2 epochs. This is due to Adam's adaptive learning rate mechanism and momentum tracking.
  * **SGD** showed slower, linear progress per epoch and required more epochs/higher learning rates to reach comparable performance.



### Task 4: Train a Neural Network & TensorBoard Analysis

#### **Setup & Logging**
* Trained a sequential neural network on MNIST for 5 epochs with a `tf.keras.callbacks.TensorBoard(log_dir='logs/fit/')` callback.
* Visualized loss and accuracy scalar curves in TensorBoard.

#### **Section 4.1: Short-Answer Analysis**

1. **What patterns do you observe in the training and validation accuracy curves?**
   * During initial epochs, both training accuracy and validation accuracy rise rapidly while training loss and validation loss decrease steadily. By epoch 5, training accuracy reaches approximately **98–99%** and validation accuracy plateaus around **97–98%**, demonstrating steady learning convergence.

2. **How can you use TensorBoard to detect overfitting?**
   * Overfitting is detected in TensorBoard under the **Scalars** tab by observing divergence between `train` and `validation` lines. Overfitting occurs when **training loss continues to decrease** and **training accuracy increases**, but **validation loss begins to rise** or **validation accuracy levels off/declines**. A widening gap between the two curves indicates memorization of training data.

3. **What happens when you increase the number of epochs?**
   * Increasing epochs provides the optimizer with more gradient update steps. In early stages, it improves model performance. However, training for an excessively large number of epochs without regularization (such as Dropout or Early Stopping) causes the network to overfit, lowering generalization accuracy on unseen test data.
