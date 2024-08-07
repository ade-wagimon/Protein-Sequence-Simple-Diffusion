# Protein Sequence Diffusion Model

This repository contains a TensorFlow implementation of a reverse diffusion model for protein sequence generation. The model takes aligned protein sequences, encodes them with various biochemical properties, and trains a LSTM-based model to generate new sequences.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Sequence Generation](#sequence-generation)
- [Plotting Training History](#plotting-training-history)

## Installation

Ensure you have the required dependencies installed:

```bash
pip install tensorflow numpy biopython scikit-learn matplotlib
```

## Usage

To run the project, follow these steps:

1. Place your FASTA alignment file in the project directory and set the `alignment_file` variable to its name.
2. Execute the script:

```bash
python script_name.py
```

Replace `script_name.py` with the name of your script file.

## Project Structure

```
.
├── Align45.fasta
├── script_name.py
└── README.md
```

- `Align45.fasta`: Your protein sequence alignment file.
- `script_name.py`: The main script for data preparation, model training, evaluation, and sequence generation.
- `README.md`: This file.

## Data Preparation

The data preparation step reads an alignment file in FASTA format and encodes each sequence with a combination of one-hot encoding and various biochemical properties (hydrophobicity, molecular weight, charge, polarity, aromaticity, and acidity/basicity).

```python
X_reshaped = prepare_data(alignment_file)
```

## Model Training

The model is an LSTM-based network that takes encoded sequences as input and attempts to predict the next sequence in the series.

```python
model, history = train_model(X_reshaped, num_steps=10, batch_size=32)
```

## Evaluation

After training, the model is evaluated using Mean Absolute Error (MAE).

```python
loss, mae = evaluate_model(model, X_reshaped)
print(f"Test MAE: {mae:.2f}")
```

## Sequence Generation

The model can generate new sequences starting from an initial sequence.

```python
initial_sequence = X_reshaped[0]
generated_sequence = generate_sequence(model, initial_sequence, steps=10)
decoded_sequence = decode_sequence(generated_sequence)
print("Decoded Sequence:", decoded_sequence)
```

## Plotting Training History

The training and validation loss over epochs can be plotted for analysis.

```python
plot_history(history)
```

## Acknowledgements

This project uses the following resources:
- TensorFlow for building and training the model.
- Biopython for sequence alignment handling.
- Scikit-learn for data scaling.
- Matplotlib for plotting training history.

## License

This project is licensed under the MIT License.
