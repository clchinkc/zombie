
import os
import tkinter as tk
from tkinter import Entry, OptionMenu, filedialog, messagebox

from PIL import Image


def select_directory():
    root = tk.Tk()
    root.withdraw()

    directory = filedialog.askdirectory(
        title = "Select a directory",
        mustexist = True
    )
    return directory

def choose_format(root, original_format, converted_format):
    tk.Label(root, text="Original format:").grid(row=0, column=0)
    original_format_entry = OptionMenu(root, original_format, "webp", "jpeg", "png", "svg")
    original_format_entry.grid(row=0, column=1)

    tk.Label(root, text="Converted format:").grid(row=1, column=0)
    converted_format_entry = OptionMenu(root, converted_format, "webp", "jpeg", "png", "svg")
    converted_format_entry.grid(row=1, column=1)

    tk.Button(root, text="Convert", command=lambda: convert_format(directory, original_format, converted_format)).grid(row=2, column=0, columnspan=2, pady=10)

def convert_format(directory, original_format, converted_format):
    for filename in os.listdir(directory):
        if filename.endswith("." + original_format.get()):
            image = Image.open(os.path.join(directory, filename))
            image.save(os.path.join(directory, filename.replace("." + original_format.get(), "." + converted_format.get())), converted_format.get().upper())
    messagebox.showinfo("Success", "All images in the selected directory have been converted to the selected format.")

root = tk.Tk()
original_format = tk.StringVar()
converted_format = tk.StringVar()
choose_format(root, original_format, converted_format)

directory = select_directory()
if directory:
    root.mainloop()
else:
    messagebox.showerror("Error", "No directory was selected.")

