
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

1. **Clone the Project Repository:**

   ```sh
   git clone https://github.com/gstiebler/llm_mri.git
   cd llm_mri
   ```

2. **Install Hugging Face Transformers CLI:**

   Before running the script, ensure you have the Hugging Face CLI installed and configured to download models. Follow the instructions at [Hugging Face CLI Guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to set up the CLI.

3. **Download the GEMM-2b Model:**

   With the Hugging Face CLI set up, download the `google/gemma-2b-it` model by running the following command:

   ```sh
   huggingface-cli repo download google/gemma-2b-it --repo-type model
   ```

   This step ensures that the required model is available locally for the analysis script.

4. **Run the Script:**

   After setting up the CLI and downloading the model, you can run the analysis script with:

   ```sh
   python activation_analysis.py
   ```

This script will execute the analysis, comparing activation patterns across different prompts, and generate the corresponding visualizations and outputs.

## Understanding the Outputs

- **Activation Patterns Visualization**: For each prompt, the script plots the activation values of the `GemmaMLP` layer. This visualization helps in understanding how the activations differ between the two prompts.

- **Correlation Plot**: A plot showing the Pearson correlation coefficients across the activations of the `GemmaMLP` layers for the two prompts. A higher correlation might indicate similar processing patterns by the model for the given prompts.

- **Console Outputs**: The script prints out the generated responses to the prompts and the correlation coefficients for each layer, providing a textual insight into the model's behavior and the similarity between the activation patterns.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests if you have suggestions for improvements or have identified issues with the current analysis.
