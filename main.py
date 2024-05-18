from transformers import BertTokenizer, BertModel
import torch
from flask import Flask, jsonify, request 

app = Flask(__name__)

@app.route('/get-similarity-score',  methods=['POST'])
def your_endpoint():
    requestData = request.json 
    print(requestData)
    resumePrompt =', '.join(requestData['resumePrompt'])
    requirementsPrompt = ', '.join(requestData['requirementsPrompt'])
    print('imported')
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    print('encoding')
    inputs1 = tokenizer(resumePrompt, return_tensors="pt")
    inputs2 = tokenizer(requirementsPrompt, return_tensors="pt")

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embeddings1 = outputs1.last_hidden_state.mean(dim=1) 
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    similarity_score = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).item()

    return jsonify({'score': similarity_score})


if __name__ == "__main__":
    app.run( host='0.0.0.0',port=8000)


