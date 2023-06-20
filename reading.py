# Required Libraries
import tkinter as tk
from tkinter import messagebox

import requests
import spacy
from googletrans import LANGUAGES, Translator


# Checking internet connectivity
def is_connected():
    try:
        requests.get('http://google.com', timeout=5)
        return True
    except:
        return False

# Text Highlighting
def highlight_text(text):
    # Load English tokenizer, POS tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Extract the important parts
    important_parts = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB']]

    # This could be further customized for highlighting
    return important_parts

# Text Translation
def translate_text(text):
    if not is_connected():
        messagebox.showerror("Network Error", "No internet connection. Check your network settings.")
        return ""
    
    # Instantiate the translator
    translator = Translator()

    # Translate the text to Chinese
    translated = translator.translate(text, dest='zh-cn')

    return translated.text

# GUI
class TextReaderApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Text Reader App")
        self.geometry("800x600")

        # Text entry
        self.text_entry = tk.Text(self, height=10)
        self.text_entry.pack()

        # Buttons
        self.highlight_button = tk.Button(self, text="Highlight", command=self.highlight)
        self.highlight_button.pack()

        self.translate_button = tk.Button(self, text="Translate", command=self.translate)
        self.translate_button.pack()

        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.clear_button.pack()

        # Result Display
        self.result_text = tk.Text(self, height=10)
        self.result_text.pack()

    def highlight(self):
        text = self.text_entry.get('1.0', 'end-1c')
        highlighted = highlight_text(text)
        self.result_text.insert('end', f"Important parts: {highlighted}")

    def translate(self):
        text = self.text_entry.get('1.0', 'end-1c')
        translation = translate_text(text)
        self.result_text.insert('end', f"Translation: {translation}")

    def clear(self):
        self.text_entry.delete('1.0', 'end')
        self.result_text.delete('1.0', 'end')


if __name__ == "__main__":
    app = TextReaderApp()
    app.mainloop()


# Highlight: only output the important parts of the text, not highlight the text itself
# Translate: with "translated text: " in front of the translated text
# bad UI
