import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

import cv2
from tkinter import *
from PIL import ImageTk, Image
import _tkinter # with underscore, and lowercase 't'

win=tk.Tk()

def b1_click():
    global path2
    try:
        json_file = open('model2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model2.h5")
        print("Loaded model from disk")
        label=['apple_pie','baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
                'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
                'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
                'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel',
                'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
                'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
                'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
                'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog',
                'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
                'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
                'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
                'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
                'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
                'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
                'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
                'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

        
        #lbl2=tk.Label(win,image=img)
        
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")
        #img1=('F:/py/leaf_disease_final( COMPLETE )/1.jpg')


        #lbl2=tk.Label(win,image=img1)
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")
        #loading image 
        path2=filedialog.askopenfilename()
        print(path2)
        

        #img = ImageTk.PhotoImage(Image.open(path2))
        
        #lbl2=tk.Label(win,image=img)
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")

        #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        #panel = tk.Label(win, image = img)
        #panel.pack( fill = "both", expand = "yes")
        #imr=cv2.imread(path2)
        #a=cv2.imshow(imr)
        #print(imr)
        test_image = image.load_img(path2, target_size = (128, 128))        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        #print(result)
        #print(result)
        fresult=np.max(result)
        label2=label[result.argmax()]
        print(label2)
        #lb2.configure(image=img)
        #lbl2.image=img
        lbl.configure(text=label2)
         
        
        #lbl2(ent.config(state='disabled'))
        win.mainloop()
        

    except IOError:
        pass


#button

#labelframe = LabelFrame(win, text="Leaf Disease Detection using OPENCV")
#labelframe.pack(fill="both", expand="yes")
label1 = Label(win, text="GUI For Food Detection using OPENCV", fg ='blue')
label1.pack()
    
b1=tk.Button(win, text= 'browse image',width=25, height=3,fg ='red', command=b1_click)
b1.pack()
lbl = Label(win, text="Result", fg ='blue')
lbl.pack()

#image =ImageTk.PhotoImage(file='a.JPG')

#img1='1.JPG'
#lb2 = Label(win,image=image)
#lb2.pack()


#lbl.grid(column=0, row=0)
win.geometry("550x250")
win.title("Food  Detection using OPENCV")
win.bind("<Return>", b1_click)
win.mainloop() 
