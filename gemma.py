from transformers import AutoTokenizer, pipeline
import torch

import matplotlib.pyplot as plt

model = "google/gemma-2b-it"

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
        tensor = output[0].detach()
        if name in activations[index]:
            activations[index][name].append(tensor)
        else:
            activations[index][name] = [tensor]
    return hook

# Step 3: Attach the hook to each layer
for name, layer in pipeline.model.named_modules():
    '''
    print("")
    print(name)
    print("------*********------")
    print(layer)
    print("------*********------")
    print("------*********------")
    print("------*********------")
    print("------*********------")
    '''
    if layer.__class__.__name__ == "GemmaMLP":
        layer.register_forward_hook(get_activation(name))

message_better = [
    {"role": "user", "content": "What emotion is better: love or hate? Print only one of these 2 words, nothing else."},
]
message_worse = [
    {"role": "user", "content": "What emotion is worse: love or hate? Print only one of these 2 words, nothing else."},
]

messages = [message_better, message_worse]

for i in range(2):
    prompt = pipeline.tokenizer.apply_chat_template(messages[i], tokenize=False, add_generation_prompt=True)
    index = i
    outputs = pipeline(
        prompt,
        max_new_tokens=5,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95
    )
    print(outputs[0]["generated_text"][len(prompt)])


'''
for layer_name, activation in activations.items():
    float_list = activation.view(-1).tolist()
    float_list.sort(reverse=True)
    plt.plot(float_list) 
    plt.xlabel('Index') 
    plt.ylabel('Value') 
    plt.title(layer_name) 
    plt.show()
'''
