
# Activation Analysis in gemma-2b Model

This project is designed to investigate and compare the activation patterns of the GEMM-2b model (referred to as "gemma" in the code), specifically within its `GemmaMLP` layers, when prompted with different inputs. By analyzing these patterns, we aim to gain insights into how different prompts can influence the internal workings of the model, potentially uncovering similarities or differences in how the model processes various types of information.

## Overview

The script uses the `transformers` library from Hugging Face to load the GEMM-2b model and defines hooks to capture the activations within the `GemmaMLP` layers. It then processes two different prompts through the model, captures the activations, and computes the Pearson correlation coefficient between the activation patterns of these prompts. Additionally, it visualizes the activation patterns and their correlations to facilitate analysis.

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- Matplotlib
- Pandas
- SciPy

Ensure all dependencies are installed using the following command:

```sh
pip install torch transformers matplotlib pandas scipy
```

## Setup and Execution

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Run the script:**

   ```sh
   python activation_analysis.py
   ```

## Understanding the Outputs

- **Activation Patterns Visualization**: For each prompt, the script plots the activation values of the `GemmaMLP` layer. This visualization helps in understanding how the activations differ between the two prompts.

- **Correlation Plot**: A plot showing the Pearson correlation coefficients across the activations of the `GemmaMLP` layers for the two prompts. A higher correlation might indicate similar processing patterns by the model for the given prompts.

- **Console Outputs**: The script prints out the generated responses to the prompts and the correlation coefficients for each layer, providing a textual insight into the model's behavior and the similarity between the activation patterns.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests if you have suggestions for improvements or have identified issues with the current analysis.
