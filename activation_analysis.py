from transformers import AutoTokenizer, pipeline
import torch

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


# Load the GEMM-2b model
model = "google/gemma-2b-it"

# Initialize the tokenizer and pipeline
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

index = -1

# Step 1: Define a container to hold the activations
activations = [{}, {}]

# Step 2: Define a hook function
def get_activation(name):
    def hook(model, input, output):
        # print("Executing hook for layer:", name)
        print("Output shape:", output.shape)
        tensor = output[0].detach()
        if name in activations[index]:
            activations[index][name].append(tensor)
        else:
            activations[index][name] = [tensor]
    return hook

# Step 3: Attach the hook to each layer
for name, layer in pipeline.model.named_modules():
    if layer.__class__.__name__ == "GemmaMLP":
        layer.register_forward_hook(get_activation(name))

messages = [
    "What's the capital of France? Print only one word, nothing else.",
    "What's the capital of Germany? Print only one word, nothing else."
]

for i in range(2):
    message = [
        {"role": "user", "content": messages[i]},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    index = i
    outputs = pipeline(
        prompt,
        max_new_tokens=2,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95
    )
    generated = outputs[0]["generated_text"]
    len_input = len(prompt)
    print(generated[len_input:])


correlations = []
for layer_name, value in activations[0].items():
    activation1 = activations[0][layer_name][1]
    activation2 = activations[1][layer_name][1]
    
    float_list1 = activation1.view(-1).tolist()
    print("Tensor size:", activation1.size())
    float_list_with_indexes = [[float_list1[i], i] for i in range(len(float_list1))]
    float_list_with_indexes.sort(key=lambda x: x[0], reverse=True)
    
    sorted_values1 = [item[0] for item in float_list_with_indexes]
    
    
    float_list2 = activation2.view(-1).tolist()
    values2 = [float_list2[item[1]] for item in float_list_with_indexes]
    
    correlation = scipy.stats.pearsonr(sorted_values1, values2)[0]
    print("Correlation:", correlation)
    correlations.append(correlation)
    
    plt.plot(sorted_values1, label='Activation 1')
    plt.plot(values2, label='Activation 2')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(layer_name)
    plt.legend()
    plt.show()

plt.plot(correlations, label='correlation')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('correlations')
plt.legend()
plt.show()