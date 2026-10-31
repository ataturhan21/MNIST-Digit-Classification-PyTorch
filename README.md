# MNIST Digit Classification with PyTorch

A notebook for Kaggle's Digit Recognizer competition. It reads labeled and unlabeled pixel CSV files, trains a fully connected PyTorch classifier, and exports submission predictions.

**[Open the notebook](mnist_digit_classification.ipynb)** · **[Dependencies](requirements.txt)** · **[MIT license](LICENSE)**

## Model and workflow

- Normalize 784 pixel values to the range 0–1.
- Split the labeled CSV into 80% training and 20% validation data with `random_state=42`.
- Train a 784 → 128 → 64 → 10 network with ReLU activations, cross-entropy loss, and Adam at a learning rate of 0.001.
- Use batches of 64 and 10 epochs in the default training call.
- Generate `submission.csv` with `ImageId` and `Label` columns.

The implementation uses the Kaggle competition CSVs rather than downloading MNIST through `torchvision.datasets`.

## Run on Kaggle

1. Create a notebook and attach the [Digit Recognizer competition data](https://www.kaggle.com/competitions/digit-recognizer/data).
2. Import `mnist_digit_classification.ipynb`.
3. Confirm the paths in **Loading the Dataset**:

   ```python
   train_data_path = "/kaggle/input/digit-recognizer/train.csv"
   test_data_path = "/kaggle/input/digit-recognizer/test.csv"
   ```

4. Run the cells in order, then download `submission.csv` from the working directory.

The current code runs on the CPU; it does not move the model or input tensors to a GPU.

## Run locally

```bash
git clone https://github.com/ataturhan21/MNIST-Digit-Classification-PyTorch.git
cd MNIST-Digit-Classification-PyTorch
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt notebook
jupyter notebook mnist_digit_classification.ipynb
```

Download the competition CSVs separately and change the 2 input paths before executing the notebook. The dataset and trained weights are not included.

## Reproducibility

The notebook seeds NumPy and PyTorch and uses a fixed validation split. Saved metadata records Python 3.10.14, but the dependency list is unpinned; record the package versions used for a new run. Seeds do not guarantee identical results across every platform or package version.

The notebook contains historical validation output. No current leaderboard score or freshly reproduced accuracy is asserted here.

## Reliability checks

Training and validation losses are weighted by the number of samples, including partial batches. Validation accuracy uses the supplied loader rather than a global dataset. Focused checks use small tensor/model doubles; they validate metric arithmetic without training PyTorch or reproducing model accuracy.

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow installs only the dependencies used by these checks and performs no model downloads or training.

## License

[MIT](LICENSE).
