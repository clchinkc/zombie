

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

"""
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

"""

# https://twitter.com/joao_gante/status/1679775399172251648
# https://huggingface.co/inference-endpoints
# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da
# An example command for fine-tuning Llama 2 7B on the timdettmers/openassistant-guanaco can be found below. The script can merge the LoRA weights into the model weights and save them as safetensor weights by providing the merge_and_push argument. This allows us to deploy our fine-tuned model after training using text-generation-inference and inference endpoints.
# python finetune_llama_v2.py \
# --model_name meta-llama/Llama-2-7b-hf \
# --dataset_name timdettmers/openassistant-guanaco \
# --use_4bit \
# --merge_and_push

# https://github.com/facebookresearch/llama-recipes/tree/main

# https://github.com/huggingface/text-generation-inference
