
"""
Reddit Reply Bot
-------------------------------------------------------------
pip install praw pyenchant
"""

"""
import praw
import enchant


def reddit_bot(sub, trigger_phrase):
   reddit = praw.Reddit(
       client_id='your_client_id',
       client_secret='your_client_secret',
       username='your_username',
       password='your_pw',
       user_agent='your_user_agent'
   )

   subreddit = reddit.subreddit(sub)
   dict_suggest = enchant.Dict('en_US')

   for comment in subreddit.stream.comments():
       if trigger_phrase in comment.body.lower():
           word = comment.body.replace(trigger_phrase, '')
           reply_text = ''
           similar_words = dict_suggest.suggest(word)
           for similar in similar_words:
               reply_text += (similar + ' ')
           comment.reply(reply_text)


if __name__ == '__main__':
   reddit_bot(sub='Python', trigger_phrase='useful bot')
"""
"""
The code is a Python project that creates an automated Reddit bot using the praw and enchant modules. The bot checks every comment in a selected subreddit and replies to any comments that contain a predefined 'trigger phrase'. The program uses the praw module to interact with Reddit's API, and enchant to generate similar words to the comment, allowing the bot to make an appropriate reply. You’ll need to check out https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps to get hold of a client_id, client_secret, username, password, and user_agent. You’ll need this information to make comments to reddits via the API interface. The code includes a function called reddit_bot that takes in the subreddit and trigger phrase as parameters and continuously checks for new comments to reply to. The bot replaces the trigger phrase with an empty string and generates similar words to create a reply.
"""

"""
Chatbot
The project is about creating an automated chatbot using Python's ChatterBot module, which uses AI to answer user's questions.
The program is relatively small, and users can explore ChatterBot's documentation and the broader field of AI chatbots to expand the code's features.
However, it is important to note that ChatterBot is no longer being actively maintained. This means you need to make a small change to the tagging.py file located in the ‘Lib/site-packages/chatterbot’ directory of your Python installation folder.

Source Code:
"""

"""
Chat Bot
-------------------------------------------------------------
1) pip install ChatterBot chatterbot-corpus spacy
2) python3 -m spacy download en_core_web_sm
   Or... choose the language you prefer
3) Navigate to your Python3 directory
4) Modify Lib/site-packages/chatterbot/tagging.py
  to properly load 'en_core_web_sm'
"""

"""
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

def create_chat_bot():
   chatbot = ChatBot('Chattering Bot')
   trainer = ChatterBotCorpusTrainer(chatbot)
   trainer.train('chatterbot.corpus.english')

   while True:
       try:
           bot_input = chatbot.get_response(input())
           print(bot_input)

       except (KeyboardInterrupt, EOFError, SystemExit):
           break


if __name__ == '__main__':
   create_chat_bot()
"""

"""
Modify tagging.py:

Find the first code snippet which is part of the __init__ method for the PosLemmaTagger class. Replace this with the if/else statement.

Note: this example is for the English library we used in our example, but feel free to switch this out to another language if you’d prefer.
"""

"""
# Replace this:
self.nlp = spacy.load(self.language.ISO_639_1.lower())

# With this:
if self.language.ISO_639_1.lower() == 'en':
   self.nlp = spacy.load('en_core_web_sm')
else:
   self.nlp = spacy.load(self.language.ISO_639_1.lower())
"""


