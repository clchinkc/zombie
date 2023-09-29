from transformers import pipeline, set_seed

# Set seed for reproducibility
set_seed(42)

# Initialize chatbot pipeline
chatbot = pipeline('text2text-generation', model='facebook/blenderbot-3B', tokenizer='facebook/blenderbot-3B')

# Start the conversation
print('Hello! I am your assistant. How can I assist you today?')
chat_history = 'Chatbot: Hello! I am your assistant. How can I assist you today?\n'

# List of questions
questions = [
    "What is ray tracing?",
    "What is the difference between ray tracing and path tracing?",
    "What is my last question?"
]

# Chatbot loop
while questions:
    # Get user input from the questions list (and remove it from the list)
    user_input = questions.pop(0)
    
    # Print the user input (so we can see the question being asked)
    print('User:', user_input)

    # Append user input to chat history
    chat_history += f'User: {user_input}\n'
    
    # Generate chatbot response based on chat history
    chatbot_response = chatbot(chat_history)[0]['generated_text']
    
    # Extract the chatbot's part of the response
    chatbot_response = chatbot_response.split('User: ')[-1].replace(f'User: {user_input}\n', '').strip()

    # Print chatbot response
    print('Chatbot:', chatbot_response)
    
    # Append chatbot response to chat history
    chat_history += f'Chatbot: {chatbot_response}\n'





def __generate(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# https://twitter.com/joao_gante/status/1679775399172251648
# https://huggingface.co/inference-endpoints
# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da


# https://github.com/facebookresearch/llama-recipes/tree/main

# https://github.com/huggingface/text-generation-inference
