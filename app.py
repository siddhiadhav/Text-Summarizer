from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

app = Flask(__name__)

# Load the BART model and tokenizer for summarization
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        inputtext = request.form["inputtext_"]
        
        # Pre-process the input text
        input_text = inputtext.strip()  # Clean the input text
        
        # Tokenize the input text
        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
        
        # Generate the summary
        summary_ids = model.generate(
            tokenized_text,
            num_beams=5,  # Increase beams for better exploration
            length_penalty=1.0,  # Adjust length penalty for better-length summaries
            max_length=300,  # Limit the max length of the summary
            min_length=50,  # Ensure the summary is not too short
            early_stopping=True  # Stop when an optimal summary is generated
        )
        
        # Decode the summary and return it
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Render the output in the template
        return render_template("output.html", data={"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)

