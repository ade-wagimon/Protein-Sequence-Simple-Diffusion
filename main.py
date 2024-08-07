import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
from Bio import AlignIO
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# File paths
alignment_file = "Align45.fasta"

# Hydrophobicity index and molecular weight tables for amino acids
hydrophobicity_index = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

molecular_weight = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15, 'Q': 146.14, 'E': 147.13, 
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 
    'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

polarity = {'A': 'N', 'R': 'P', 'N': 'P', 'D': 'P', 'C': 'N', 'Q': 'P', 'E': 'P', 
             'G': 'N', 'H': 'P', 'I': 'N', 'L': 'N', 'K': 'P', 'M': 'N', 'F': 'N', 
             'P': 'N', 'S': 'P', 'T': 'P', 'W': 'N', 'Y': 'P', 'V': 'N'}

aromaticity = {'A': 'N', 'R': 'N', 'N': 'N', 'D': 'N', 'C': 'N', 'Q': 'N', 'E': 'N', 
               'G': 'N', 'H': 'N', 'I': 'N', 'L': 'N', 'K': 'N', 'M': 'N', 'F': 'Y', 
               'P': 'N', 'S': 'N', 'T': 'N', 'W': 'Y', 'Y': 'Y', 'V': 'N'}

acidity_basicity = {'A': 4.06, 'R': 12.48, 'N': 10.70, 'D': 3.86, 'C': 8.33, 'Q': 10.53, 'E': 4.26, 
                    'G': 5.97, 'H': 6.00, 'I': 6.02, 'L': 6.00, 'K': 10.54, 'M': 10.07, 'F': 3.90, 
                    'P': 10.47, 'S': 9.15, 'T': 9.10, 'W': 3.82, 'Y': 10.09, 'V': 7.39}

charge_info = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 
               'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 
               'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}



def encode_sequence(sequence):
    amino_acid_order = 'ARNDCQEGHILKMFPSTWYV'
    one_hot = np.zeros((len(sequence), len(amino_acid_order)))
    hydrophobicity = np.zeros(len(sequence))
    molecular_weight_array = np.zeros(len(sequence))
    charge_array = np.zeros(len(sequence))
    polarity_array = np.zeros(len(sequence))
    aromaticity_array = np.zeros(len(sequence))
    acidity_basicity_array = np.zeros(len(sequence))
    
    for i, aa in enumerate(sequence):
        if aa in amino_acid_order:
            one_hot[i, amino_acid_order.index(aa)] = 1
            hydrophobicity[i] = hydrophobicity_index.get(aa, 0)
            molecular_weight_array[i] = molecular_weight.get(aa, 0)
            charge_array[i] = charge_info.get(aa, 0)
            polarity_array[i] = polarity.get(aa, 'N') == 'P'
            aromaticity_array[i] = aromaticity.get(aa, 'N') == 'Y'
            acidity_basicity_array[i] = acidity_basicity.get(aa, 0)
    
    features = np.concatenate([
        one_hot, 
        hydrophobicity[:, np.newaxis], 
        molecular_weight_array[:, np.newaxis], 
        charge_array[:, np.newaxis], 
        polarity_array[:, np.newaxis], 
        aromaticity_array[:, np.newaxis], 
        acidity_basicity_array[:, np.newaxis]
    ], axis=1)
    
    return features


# Function to read alignment and encode sequences
def prepare_data(alignment_file):
    alignment = AlignIO.read(alignment_file, "fasta")
    X_encoded = np.array([encode_sequence(str(record.seq)) for record in alignment], dtype=np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded.reshape(X_encoded.shape[0], -1))
    X_reshaped = X_scaled.reshape(X_encoded.shape[0], X_encoded.shape[1], X_encoded.shape[2])
    return X_reshaped

# Model definition
def reverse_diffusion_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(128, return_sequences=True)(x)  # Ensure LSTM returns sequences
    outputs = Dense(input_shape[-1])(x)  # Output shape matches input shape
    return Model(inputs, outputs)

# Function to compile and train model
def train_model(X_reshaped, num_steps=10, batch_size=64):
    input_shape = X_reshaped.shape[1:]
    model = reverse_diffusion_model(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = None
    for step in range(num_steps):
        X_noisy = X_reshaped  # Simulating noisy data (unsupervised setup)
        history_step = model.fit(X_noisy, X_reshaped, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=2)
        if history is None:
            history = history_step.history
        else:
            for key in history.keys():
                history[key].extend(history_step.history[key])
    
    return model, history

# Function to evaluate model
def evaluate_model(model, X_reshaped):
    loss, mae = model.evaluate(X_reshaped, X_reshaped)
    print(f"Test MAE: {mae:.2f}")
    return loss, mae

# Function to plot training history
def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to generate sequence
def generate_sequence(model, initial_sequence, steps=100):
    current_sequence = np.array([initial_sequence])  # Initial sequence to start generation
    generated_sequence = []

    for _ in range(steps):
        next_step = model.predict(current_sequence)
        generated_sequence.append(next_step[0])
        current_sequence = np.append(current_sequence[:, 1:, :], next_step[:, -1:, :], axis=1)

    # Flatten the list of arrays into a single array
    generated_sequence = np.concatenate(generated_sequence, axis=0)

    return generated_sequence


# Function to decode sequence
def decode_sequence(encoded_sequence):
    decoded_sequence = []
    amino_acid_order = 'ARNDCQEGHILKMFPSTWYV'
        
    for step in encoded_sequence:
        idx = np.argmax(step[:len(amino_acid_order)])  # Consider only the amino acid part
        if idx < len(amino_acid_order):
            amino_acid = amino_acid_order[idx]
            decoded_sequence.append(amino_acid)
        else:
            decoded_sequence.append('X')  # Placeholder for unknown amino acid
    
    return ''.join(decoded_sequence)

# Main execution
if __name__ == "__main__":
    X_reshaped = prepare_data(alignment_file)
    
    model, history = train_model(X_reshaped, num_steps=10, batch_size=32)
    
    evaluate_model(model, X_reshaped)
    
    plot_history(history)
    
    initial_sequence = X_reshaped[0]
    generated_sequence = generate_sequence(model, initial_sequence, steps=10)
    
    print("Generated Sequence Shape:", generated_sequence.shape)
    
    decoded_sequence = decode_sequence(generated_sequence)
    
    print("Decoded Sequence:")
    print(decoded_sequence)
