from transformers import BertTokenizer, BertModel
import torch
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/get-similarity-score')
def your_endpoint():
    
    print('imported')
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    print('encoding')
    sentence1 = "intern"
    sentence2 = "highly experienced"
    inputs1 = tokenizer(sentence1, return_tensors="pt")
    inputs2 = tokenizer(sentence2, return_tensors="pt")

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embeddings1 = outputs1.last_hidden_state.mean(dim=1) 
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    similarity_score = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).item()

    return jsonify({'score': similarity_score})


if __name__ == "__main__":
    app.run(port=8000,debug=True)


