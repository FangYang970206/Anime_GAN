import torch
import tkinter as tk
import os 
import numpy as np
from tkinter import ttk
import scipy.misc
from PIL import Image,ImageTk


win = tk.Tk()
win.title('Conditional-GAN-GUI')
win.geometry('200x200')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generate = torch.load("save/generate.t7").to(device)
generate.eval()

def create():
    z = np.random.normal(0, np.exp(-1 / np.pi), [1, 62])
    line = comboxlist1.get() + ' ' + comboxlist2.get()
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    y = np.zeros((1, len(tag_dict)))

    for i in range(len(tag_dict)):
        if tag_dict[i] in line:
            y[0][i] = 1
    
    image = generate(torch.from_numpy(z).float().to(device), torch.from_numpy(y).float().to(device)).to("cpu").detach().numpy()
    image = np.squeeze(image)
    image = image.transpose(1, 2, 0)
    scipy.misc.imsave('anime.png', image)
    img_open = Image.open('anime.png')
    img = ImageTk.PhotoImage(img_open)
    label.configure(image=img)
    label.image=img
    

# def go(*args):   #处理事件，*args表示可变参数
#     print(comboxlist.get()) #打印选中的值
 
comvalue1=tk.StringVar()#窗体自带的文本，新建一个值
comboxlist1=ttk.Combobox(win,textvariable=comvalue1) #初始化
comboxlist1["values"]=('orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 
                      'purple hair', 'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair')
comboxlist1.current(0)  #选择第一个
# comboxlist.bind("<<ComboboxSelected>>",go)  #绑定事件,(下拉列表框被选中时，绑定go()函数)
comboxlist1.pack()
 
comvalue2=tk.StringVar()
comboxlist2=ttk.Combobox(win,textvariable=comvalue2)
comboxlist2["values"]=('gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                      'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes')
comboxlist2.current(0)
# comboxlist.bind("<<ComboboxSelected>>",go)
comboxlist2.pack()

bm = tk.PhotoImage(file ='anime.png')
label = tk.Label(win, image = bm)
label.pack()

b = tk.Button(win,
    text='create',      # 显示在按钮上的文字
    width=15, height=2, 
    command=create)     # 点击按钮式执行的命令
b.pack()   # 按钮位置

win.mainloop()