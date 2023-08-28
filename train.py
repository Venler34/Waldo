import model
import torch
from torch import nn, load, save
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A
from dataset import WDataset
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToPILImage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 16

# train_transform = A.Compose(
#     A.Rotate(),
#     A.HorizontalFlip(),
#     A.VerticalFlip(),
#     A.Normalize(
#         mean=[0.0,0.0,0.0],
#         std=[1.0,1.0,1.0],
#         max_pixel_value=255.0
#     )
# )

myDataset = WDataset()
myModel = model.UNET().to(DEVICE)
loader = DataLoader(myDataset, BATCH_SIZE) # dataset, batch size
loss_fn = nn.BCEWithLogitsLoss() # Binary cross entropy loss
adam = Adam(myModel.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

#one training step
def train():
    for batch in tqdm(loader):
        images, masks = batch
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # autocast helps transfer from float32 to float16 or int values reduce memory usage
        with torch.cuda.amp.autocast():
            predicted = myModel(images)
            loss = loss_fn(predicted, masks)

        adam.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(adam)
        scaler.update()
    
def getData():
    background = Image.open("nowaldo.png")
    waldo = Image.open("Waldo.png")

    #Resize waldo and background
    #337 by 400
    scale = 20/400

    waldoWidth = int(waldo.width * scale)
    waldoHeight = int(waldo.height * scale)

    waldo = waldo.resize((waldoWidth,waldoHeight))
    background = background.resize((400,400))

    withoutWaldo = background.convert("RGB")
    withoutWaldo = np.asarray(withoutWaldo).astype('uint8')
    withoutWaldoTesnor = to_tensor(withoutWaldo)
    withoutWaldoTensor = withoutWaldoTesnor.reshape(3,400,400)

    withWaldo = background.copy()

    row = 50
    col = 200

    withWaldo.paste(waldo, (col, row), mask=waldo)
    withWaldo = withWaldo.convert("RGB")

    withWaldo.show()

    withWaldoArr = np.asarray(withWaldo).astype('uint8')
    
    withWaldoTensor = to_tensor(withWaldoArr)

    return withoutWaldoTensor, withWaldoTensor.reshape(3, 400, 400)

# Only hold onto 2nd largest area
def nonMaxSupression(image):
    #image is pil image
    pixels = image.load()
    height = image.size[0]
    width = image.size[1]
    threshold = 0.5

    visited = [[False] * width] * height
    finalArea = []
    largestArea = -1
    for row in range(height):
        for col in range(width):
            if(not visited[row][col]):
                potentialAreaMap = [[False] * width] * height
                area = getArea(row,col,width,height,visited,pixels, threshold)
                if largestArea == -1 or area > largestArea:
                    largestArea = area
                    potentialAreaMap = [[False] * width] * height
                    potentialAreaMap = getMap(row, col, width, height, potentialAreaMap, pixels, threshold)
                    finalArea = potentialAreaMap

            
    result = Image.new(mode="L",size=(width,height), color=255)
    resultPixels = result.load()
    print(f"Largest Area is {largestArea}")
    
    for i in range(len(finalArea)):
        for j in range(len(finalArea[i])):
            resultPixels[i,j] = 0 if (finalArea[i][j]) else 255

    return result

        
def getArea(row, col, width, height, visited, pixels, threshold):
    if(row < 0 or row >= height or col < 0 or col >= width):
        return 0
    if(visited[row][col] or pixels[row,col] < threshold):
       return 0
    #must be greater than threshold or more white
    visited[row][col] = True

    left = getArea(row,col-1, width, height, visited, pixels, threshold)
    right = getArea(row, col+1, width, height, visited, pixels, threshold)
    up = getArea(row-1,col, width, height,visited, pixels, threshold)
    down = getArea(row+1, col, width, height, visited, pixels, threshold)

    return up + down + right + left + 1

def getMap(row, col, width, height, visited, pixels, threshold):
    if(row < 0 or row >= height or col < 0 or col >= width):
        return
    if(visited[row][col] or pixels[row,col] < threshold):
       return
    
    visited[row][col] = True

    getMap(row,col-1, width, height, visited, pixels, threshold)
    getMap(row, col+1, width, height, visited, pixels, threshold)
    getMap(row-1,col, width, height,visited, pixels, threshold)
    getMap(row+1, col, width, height, visited, pixels, threshold)

    return visited

# find the mean of region and get rid of small pixels
def removeSmallRegions(image):
    regionWidth = 20
    regionHeight = 20
    height = image.size[0]
    width = image.size[1]

    result = Image.new(mode="L", size=(width,height))
    resultPixels = result.load()

    pixels = image.load()
    for i in range(0, height, regionHeight):
        for j in range(0, width, regionWidth):
            sum = 0
            for row in range(i, regionHeight+i, 1):
                for col in range(j, regionWidth+j, 1):
                    sum += pixels[row,col]

            mean = sum / (regionHeight*regionWidth)
            
            for row in range(i, regionHeight+i):
                for col in range(j, regionWidth+j):
                    resultPixels[row,col] = 0 if pixels[row,col] < mean else 255
            
    return result


def getMean(image):
    pixels = image.load()
    height = image.size[0]
    width = image.size[1]
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += pixels[i,j]
    
    return sum / (width * height)

def findDenseAreas(image):
    width = 40
    height = 40
    imageWidth = 400
    imageHeight = 400

    pixels = image.load()

    arr = []

    for i in range(0, imageHeight, height):
        arrRow = []
        for j in range(0, imageWidth, width):
            sum = 0
            for row in range(i, i + height):
                for col in range(j, j + width):
                    sum = pixels[row,col]
            arrRow.append(sum)
            # print(sum)
        arr.append(arrRow)
    
    #Find top ten value
    topSpots = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            topSpots.append((i,j, arr[i][j]))
        
    topSpots.sort(reverse=False, key=compareThird)
    topSpots = topSpots[0:20]
    print(topSpots[0])
    print(topSpots[-1])

    resultImage = Image.new(mode="L", size=(400,400), color=255)
    resultPixels = resultImage.load()

    for tupleSpot in topSpots:
        row, col, _ = tupleSpot
        row *= height
        col *= width
        
        for i in range(row, row + height):
            for j in range(col, col + width):
                resultPixels[i,j] = pixels[i,j]

    return resultImage

def compareThird(elem):
    return elem[2]

def setColor(image, threshold):
    pixels = image.load()

    width = image.size[0]
    height = image.size[1]

    result = Image.new(mode="L", size=(width,height), color=255)
    resultPixels = result.load()

    for i in range(height):
        for j in range(width):
            if pixels[i,j] < threshold:
                resultPixels[i,j] = 0

    return result

def findLargestArea(image):
    pixels = image.load()
    width = image.size[0]
    height = image.size[1]

    visited = [[False] * width] * height

    largestRow = 0
    largestCol = 0
    largestArea = 0

    for row in range(height):
        for col in range(width):
            area = getLargestArea(row,col, width, height, visited, pixels)
            if area > largestArea:
                largestRow = row
                largestCol = col
                largestArea = area

    print(largestRow, largestCol, largestArea)

    return largestRow, largestCol, largestArea

def getLargestArea(row, col, width, height, visited, pixels):
    if(row < 0 or row >= height or col < 0 or col >= width):
        return 0
    if(visited[row][col] or pixels[col,row] != 255):
       return 0
    
    visited[row][col] = True

    left = getLargestArea(row,col-1, width, height, visited, pixels)
    right = getLargestArea(row, col+1, width, height, visited, pixels)
    up = getLargestArea(row-1,col, width, height,visited, pixels)
    down = getLargestArea(row+1, col, width, height, visited, pixels)

    return up + down + right + left + 1

# def findDifference(withWaldo, withoutWaldo):
#     tensor1 = withWaldo[0]
#     tensor2 = withoutWaldo[0]

#     for t1Row, t2Row in zip(tensor1, tensor2):
#         for t1Val, t2Val in zip(t1Row, t2Row):


if __name__ == "__main__":
    # indexNum = 1

    # with open('model_state_' + str(indexNum) + '.pt', "rb") as f:
    #     myModel.load_state_dict(load(f))

    # for epoch in range(NUM_EPOCHS):
    #     indexNum += 1
    #     train()

    #     with open('model_state_' + str(indexNum) + '.pt', "wb") as f:
    #         save(myModel.state_dict(), f)


    with open('model_state.pt', "rb") as f:
        myModel.load_state_dict(load(f))
        
    withoutWaldo, withWaldo = getData()
    withWaldo = withWaldo.unsqueeze(0).to(DEVICE)
    predict = myModel(withWaldo)
    predict = predict.squeeze(0)

    # withoutWaldo = withoutWaldo.unsqueeze(0).to(DEVICE)
    # predictWithoutWaldo = myModel(withoutWaldo)
    # predictWithoutWaldo = predictWithoutWaldo.squeeze(0)

    # predict = torch.subtract(predictWithoutWaldo, predict)
    # predict = torch.add(predict, predictWithoutWaldo)
    # predict = torch.add(predict, predictWithoutWaldo)
    # predict = torch.add(predict, predictWithoutWaldo)



    transform = ToPILImage()
    result = transform(predict)
    # result = setColor(result, 180)
    # findLargestArea(result)
    result.show()
    # # result = findDenseAreas(result)
    # result.show()

    # findDifference(predict, predictWithoutWaldo)

    # resultWithout = transform(predictWithoutWaldo)
    # resultWithout.show()

    # for i in range(10):
    #     result = removeSmallRegions(result)

    # result = nonMaxSupression(result)
    # result.show()