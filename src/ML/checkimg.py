import torch
from torchvision import transforms
from PIL import Image
import config
import sys

class CheckImage():
	def __init__(self):
		self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		model_path = config.model_path
		self.MODEL = torch.load(model_path, map_location=self.DEVICE)

		self.TRANSFORMS = transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		
		self.CLASSES = [
			'Hyundai Solaris sedan', 
			'KIA Rio sedan', 
			'SKODA OCTAVIA sedan',
		    'Volkswagen Polo sedan', 
		    'Volkswagen Tiguan'
	    ]
    

	def preprocess_img(self, img_path):
		img = Image.open(img_path).convert('RGB')
		img.load()
		return self.TRANSFORMS(img)
		

	def pred_image(self, img_path):
		inputs = self.preprocess_img(img_path)
		inputs = inputs.to(self.DEVICE)

		self.MODEL.eval()

		pred = self.MODEL(inputs.unsqueeze(0))
		
		argmax = pred.argmax(1)
		
		print(self.CLASSES[argmax])

		self.MODEL.train()

	
	
if __name__ == '__main__':
	chim = CheckImage()
	chim.pred_image(sys.argv[1])