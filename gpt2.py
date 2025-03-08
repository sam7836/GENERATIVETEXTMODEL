from transformers import pipeline

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = "Artificial Intelligence is"
generated_text = generator(prompt, max_length=100, truncation=True, num_return_sequences=1)



print(generated_text[0]['generated_text'])