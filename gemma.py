from transformers import AutoTokenizer, pipeline
import torch

model = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)



# Step 1: Define a container to hold the activations
activations = {}

# Step 2: Define a hook function
def hook_fn(m, i, o):
  activations[m] = o.detach()

# Step 3: Attach the hook to each layer
for name, layer in pipeline.model.named_modules():
  layer.register_forward_hook(hook_fn)

'''
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook
  
# Assuming you want to attach hooks to all transformer layers
for i, layer in enumerate(pipeline.model.transformer.h):
    layer.register_forward_hook(get_activation(f"layer_{i}"))
'''

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(outputs[0]["generated_text"][len(prompt):])
