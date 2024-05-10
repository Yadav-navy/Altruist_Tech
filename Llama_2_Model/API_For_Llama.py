from flask import Flask, request, jsonify

app = Flask(__name__)

# Importing your function
import torch
import torch.nn as nn
import accelerate
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",device_map='auto',use_auth_token='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')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def get_llama2_response(prompt, max_new_tokens=50):
    st=time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature= 0.00001)
    #outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=2.5, top_p= 1,repetition_penalty= 1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    et=time.time()
    time_taken=et-st
    return response,time_taken

# Define a route to receive requests with two parameters
@app.route('/api', methods=['POST'])
def api():
    #Get data from request
    data = request.get_json()

    # Extract prompt and max_new_tokens from the request data
    prompt = data.get('prompt', '')  # Get the value of 'prompt' parameter, defaulting to an empty string
    max_new_tokens = data.get('max_new_tokens', 50)  # Get the value of 'max_new_tokens' parameter, defaulting to 50 if not provided

    # Call your function with the provided inputs
    response,time_taken = get_llama2_response(prompt, max_new_tokens)

    # Return the response as JSON
    return jsonify({'response': response, 'time_taken_in_sec':time_taken})



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=3000)