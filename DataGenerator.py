from PIL import Image
import numpy as np
from random import randint

def generate_image():
    background = Image.open("nowaldo.png")
    waldo = Image.open("Waldo.png")

    background.resize((400,400))

    backgroundWithMask = background.copy()
    #337 by 400
    scale = 40/337

    waldoWidth = int(waldo.width * scale)
    waldoHeight = int(waldo.height * scale)

    row = randint(waldoHeight, background.height - waldoHeight)
    col = randint(waldoWidth, background.width - waldoWidth)
    print(row, " ", col)

    waldo = waldo.resize((waldoWidth,waldoHeight))
    background.paste(waldo, (col,row), mask=waldo)
    background = background.convert("RGB")
    background.show()

    #This creates image with mask
    pixels = waldo.load()
    for i in range(waldo.size[0]):
        for j in range(waldo.size[1]):
            if pixels[i,j] != (0,0,0,0):
                pixels[i,j] = (0, 0, 0)#complete black

    waldo.show()
    
    backgroundWithMask.paste(waldo,(col, row), mask=waldo)
    # backgroundWithMask = backgroundWithMask.convert("L")
    backgroundWithMask.show()



    #return np.asarray(background).astype('uint8'), np.asarray(backgroundWithMask).astype('uint8')

if __name__ == "__main__":
    # generate_image()

    index = 1
    with open('model_state_' + str(index) + '.pt', 'rb') as f:
        print(type(f))
