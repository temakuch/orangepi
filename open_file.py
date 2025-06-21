from tkinter import *
from tkinter import filedialog
filename = None
class FilePathWindow(Tk):
    def __init__(self):
        super().__init__()
        #self.filename = None
        self.geometry("300x300")
        self.label = Label(text="Шлях до файлу:")
        self.label.pack()
        self.button1 = Button(text = "Вибрати файл", command=self.ask_open, font=("Arial", 18))
        self.button1.pack(pady=30)
        self.button2 = Button(text = "Почати", command=self.start, width=12, font=("Arial", 18))
        self.button2.pack()
    
    def ask_open(self):
        global filename
        filename = filedialog.askopenfilename()
        self.label["text"] = f"Шлях до файлу: {filename}"  
    
    def start (self):
        self.destroy()