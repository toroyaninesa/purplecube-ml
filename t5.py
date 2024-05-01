from transformers import BertTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch


# Load pre-trained BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained T5 model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Example input text
input_text = "create rejection message for work"

# Tokenize and encode input text with BERT
input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')

# Generate text with T5 using BERT-encoded input
with torch.no_grad():
    t5_output = t5_model.generate(input_ids=input_ids)

# Decode and print generated text
generated_text = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
