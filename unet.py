import os,sys,cv2,torch,time
from torch.utils.data import Dataset
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLUm
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.image as mpimg
DATASET_PATH = os.path.join("dataset", "train")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
TEST_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
THRESHOLD = 0.5
BASE_OUTPUT = "output"
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		return len(self.imagePaths)
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)
        	if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)
		return (image, mask)

class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		return self.conv2(self.relu(self.conv1(x)))
class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	def forward(self, x):

		blockOutputs = []
		
		for block in self.encBlocks:
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		return blockOutputs
class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		for i in range(len(self.channels) - 1):
			x = self.upconvs[i](x)
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		return x
	def crop(self, encFeatures, x):
	
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		
		return encFeatures
class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=1, retainDim=True,
		 outSize=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
		super().__init__()
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
    def forward(self, x):
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		map = self.head(decFeatures)
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		return map
imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
split = train_test_split(imagePaths, maskPaths,
	test_size=TEST_SPLIT, random_state=42)
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT,
		INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
	num_workers=os.cpu_count())unet = UNet().to(DEVICE)
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=INIT_LR)
trainSteps = len(trainDS) // BATCH_SIZE
testSteps = len(testDS) // BATCH_SIZE

H = {"train_loss": [], "test_loss": [] 
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
	unet.train()
	totalTrainLoss = 0
	totalTestLoss = 0
	for (i, (x, y)) in enumerate(trainLoader):
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
		
		pred = unet(x)
		loss = lossFunc(pred, y)
		opt.zero_grad()
		loss.backward()
		opt.step()
		totalTrainLoss += loss

	with torch.no_grad():
		unet.eval()
		for (x, y) in testLoader:
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps

	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().n
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime)
    
def make_predictions(model, imagePath):
	
	model.eval()
	with torch.no_grad():
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(MASK_DATASET_PATH,
			filename)
		
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,
			INPUT_IMAGE_HEIGHT))
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		predMask = (predMask > THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
        return predMask
        

print("[INFO] loading up test image paths...")
unet = torch.load(MODEL_PATH).to(DEVICE)
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        exten = filename[filename.rfind("."):]
        fn = os.path.join(root, filename)
     	predMask= make_predictions(unet, fn)
        mpimg.imsave('pred_'+filename, predMask)



'''
print("[INFO] load up model...")
imagePaths = open(TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)]
for path in imagePaths:
	predMask,gtMask= make_predictions(unet, path)
    filename=path.split("\\").[-1]
'''    

