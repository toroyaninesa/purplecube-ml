import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# Define a function to generate rejection message for a person
def generate_rejection_message(person_name, model, tokenizer, max_length=50):
    # Define prompt
    prompt = f"Create job rejection message for person {person_name} and the reason is they have 2 years of experience"

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Decode and return generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example: Generate job rejection message for person Anna Makarova
person_name = "Anna Makarova"
rejection_message = generate_rejection_message(person_name, model, tokenizer)

print("Rejection Message for", person_name + ":", rejection_message)

