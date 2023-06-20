
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

model_name = 'google/flan-t5-large' # sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

def generate(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    for line in sys.stdin:
        print(generate(line.strip()))
        
if __name__ == '__main__':
    main()


"""
from transformers import pipeline, set_seed

# Set seed for reproducibility
set_seed(42)

# Initialize chatbot pipeline
chatbot = pipeline('text2text-generation', model='facebook/blenderbot-3B', tokenizer='facebook/blenderbot-3B')

# Start the conversation
print('Hello! I am your chatbot. How can I assist you today?')
chat_history = ''

while True:
    # Get user input
    user_input = input('User: ')
    
    # Append user input to chat history
    chat_history += f'User: {user_input}\n'
    
    # Generate chatbot response based on chat history
    chatbot_response = chatbot(chat_history, max_length=50, do_sample=True)[0]['generated_text']
    
    # Print chatbot response
    print('Chatbot:', chatbot_response)
    
    # Append chatbot response to chat history
    chat_history += f'Chatbot: {chatbot_response}\n'

"""