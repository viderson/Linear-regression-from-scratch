
# 📈 Linear Regression from Scratch in Python

This project demonstrates a full implementation of **Linear Regression** using only **NumPy**, written entirely from scratch — without relying on libraries like `scikit-learn`. It includes:

- Preprocessing and standardization
- A custom linear regression class with gradient descent
- Evaluation metrics: **MSE**, **RMSE**, and **R²**
- Interactive cost plotting with Plotly

---

## 📁 Project Structure

```
linear-regression/
├── data/
│   └── test.csv           # Sample input dataset
├── notebooks/
│   └── Linear_regression_from_scratch.ipynb
├── model/
│   └── linear_regression.py  # The model implementation (optional extraction)
└── README.md
```

---

## 🔍 Problem Statement

We aim to fit a simple linear regression model:
```
f(x) = wx + b
```
Given input features `x` and labels `y`, the model learns the optimal parameters `w` and `b` using **gradient descent** to minimize the **Mean Squared Error (MSE)**.

---

## 🧠 Key Concepts

### ✅ Forward Pass
Predicts:
```
ŷ = X · w + b
```

### ❌ Loss Function
Mean Squared Error:
```
J(w, b) = (1/2m) Σ (ŷ - y)²
```

### 🔁 Backward Pass
Gradient computation:
```
∂J/∂w = (1/m) Σ (ŷ - y) * x
∂J/∂b = (1/m) Σ (ŷ - y)
```

### 🔧 Parameter Update
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

---

## 🚀 Usage

### 1. Upload Your Data
Ensure `test.csv` contains columns `x` and `y`.

```bash
├── data/
│   └── test.csv
```

### 2. Run the notebook
Open and run:  
[`Linear_regression_from_scratch.ipynb`](notebooks/Linear_regression_from_scratch.ipynb)

### 3. Train the Model

```python
model = Linear_Regresion(learning_rate=0.01)
model.fit(X_train, y_train, iterations=10000)
```

### 4. Predict and Evaluate

```python
y_pred = model.predict(X_test)

mse = Mse(y_test, y_pred)
rmse = Rmse(y_test, y_pred)
r2 = R2(y_test, y_pred)
```

---

## 📊 Evaluation Metrics

| Metric | Meaning | Formula |
|--------|---------|---------|
| **MSE**  | Average squared error | ![MSE](https://latex.codecogs.com/svg.image?\text{MSE}=\frac{1}{n}\sum(y_{\text{true}}-y_{\text{pred}})^2) |
| **RMSE** | Square root of MSE | ![RMSE](https://latex.codecogs.com/svg.image?\text{RMSE}=\sqrt{\text{MSE}}) |
| **R²**   | Explained variance | ![R2](https://latex.codecogs.com/svg.image?R^2=1-\frac{SSR}{SST}) |

---

## 📉 Sample Output

- ✅ Cost convergence plot
- ✅ Model coefficients `w`, `b`
- ✅ Metrics:
  - `MSE: 0.0213`
  - `RMSE: 0.146`
  - `R²: 0.97`

---

## 💾 Model Persistence

```python
# Save model
model.save_model("linear_model.pkl")

# Load model
model.load_model("linear_model.pkl")
```

---

## 🔧 Dependencies

- Python ≥ 3.7
- NumPy
- Pandas
- Plotly
- Google Colab (for `files.upload()`)

---

## 🧪 Example Dataset Format

```csv
x,y
1.1,2.3
2.0,4.1
3.5,7.2
...
```

---

## 📚 Future Improvements

- Multi-feature support
- Batch size (mini-batch gradient descent)
- Regularization (L1/L2)
- Scikit-learn benchmark comparison

---

## 🤖 Author

**J.K.** — Machine Learning enthusiast & student of Applied Computer Science at UJ  
🌐 Check out more: [YourGitHubProfile](https://github.com/yourusername)
