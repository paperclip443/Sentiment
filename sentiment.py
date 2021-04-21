import load_pickles as s
from tkinter import *

root = Tk()
root.title('Your Text Here')
root.geometry('600x200')
e = Entry(root, width=50)
e.pack()

e.focus_set()

def take_input():
    Output.delete('1.0', END)
    if s.sentiment(e.get())[0] == 'pos':
        Output.insert(END, 'This text has positive sentiment')
    else:
        Output.insert(END, 'This text has negative sentiment')
    Output.insert(END, f'\nThe confidence of the analysis is {s.sentiment(e.get())[1]*100}%')
    # e.get() - This is the text you may want to use later

Output = Text(root, height=5, width=50)

b0 = Button(root, text = 'Assess', width = 10, command = take_input)
b0.pack()

b1 = Button(root, text='Quit', width=10, command = root.destroy)
b1.pack()

Output.pack()
mainloop()

#print(s.sentiment('This is a mixed review. Good bad awful amazing terrible outstanding great, wonderful, excellent'))