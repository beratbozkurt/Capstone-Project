import tkinter 
import pickle
from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
from sklearn import metrics


app = tkinter.Tk()
app.title("Classifier")

model = load_model("LSTM_Model")
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
classes=['BladderCancer', 'Leukemia', 'NonHodgkinLymphoma', 'ThyroidCancer']
app.geometry("800x600")


def modelling():
        result_entry.delete("1.0", "end") 
        sample_sequence = tokenizer.texts_to_sequences([abstract_entry.get("1.0", "end-1c")])
        sample_padded = pad_sequences(sample_sequence, padding="post", maxlen=200)
        predict_x=model.predict(sample_padded)
        classes_x=np.argmax(predict_x,axis=1)
        result_entry.insert("1.0", "%s with the percentage of %f " % (classes[classes_x[0]],100*predict_x[0][classes_x[0]])) 

abstract_label = tkinter.Label(app,text="The Abstract",bg="#FFFFFF") 
abstract_label.place(x=10,y=10)

result_label = tkinter.Label(app,text="Class",bg="#FFFFFF") 
result_label.place(x=10,y=465)

abstract_entry = tkinter.Text(app, height = 25,width = 98,bg = "light gray")
abstract_entry.place(x=4, y=50)

result_entry = tkinter.Text(app,height = 5,width = 51,bg = "light gray",fg="purple")
result_entry.place(x= 4, y=500)
modelling_button = tkinter.Button(app, command=modelling, text="Modelling", bg="light cyan", fg="black")
modelling_button.place(x=575, y=525)
app.mainloop()