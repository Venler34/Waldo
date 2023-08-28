import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor

class WDataset(Dataset):
    def __init__(self, transform = None):
        self.defaultBackground = Image.open("nowaldo.png")
        self.waldo = Image.open("Waldo.png")
        self.transform = transform

        #Resize waldo and background
        #337 by 400 so waldo is 16 by 35
        scale = 20/400

        waldoWidth = int(self.waldo.width * scale)
        waldoHeight = int(self.waldo.height * scale)

        self.waldo = self.waldo.resize((waldoWidth,waldoHeight))
        self.defaultBackground = self.defaultBackground.resize((400,400))

        #This creates image with mask
        self.maskWaldo = self.waldo.copy()
        pixels = self.maskWaldo.load()
        for i in range(self.maskWaldo.size[0]):
            for j in range(self.maskWaldo.size[1]):
                if pixels[i,j] != (0,0,0,0):
                    pixels[i,j] = (0, 0, 0)#completely black

    def generate_image(self, row, col):
        background = self.defaultBackground.copy()
        backgroundWithMask = self.generate_mask(row,col)
        
        background.paste(self.waldo, (col,row), mask=self.waldo)

        background = background.convert('RGB')
        backgroundWithMask = backgroundWithMask.convert("L")
        # background.show()
        # backgroundWithMask.show()

        backgroundArr = np.asarray(background).astype('uint8')
        maskArr = np.asarray(backgroundWithMask).astype('uint8')

        #convert to tensor and return
        backgroundTensor = to_tensor(backgroundArr)
        maskTensor = to_tensor(maskArr)

        return backgroundTensor, maskTensor

    def generate_mask(self, row, col):
        backgroundWithMask = Image.new("RGB", size=self.defaultBackground.size, color="white")
        backgroundWithMask.paste(self.maskWaldo,(col,row),mask=self.maskWaldo)
        return backgroundWithMask



    #fake length because dataset could be infinitely long
    #but set to 1600 because only 40 * 40 spots to put waldo
    def __len__(self):
        return 1600
    
    def __getitem__(self, index):
        #use index as a row and column
        col = (index * 10) % self.defaultBackground.width
        row = ((index * 10) // self.defaultBackground.width) % self.defaultBackground.height

        image, mask = self.generate_image(row, col)

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        image = image.reshape(3, 400, 400)
        mask = mask.reshape(1, 400, 400)

        return image, mask
    
if __name__ == "__main__":
    set = WDataset()
    set.generate_image(200,200)

        