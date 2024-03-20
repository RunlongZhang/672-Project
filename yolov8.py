#prechecks (run these in terminal while in working directory of code)
#pip install ultralytics==8.0.196
#pip install ipython
#pip install roboflow
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#pip install torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
#mkdir datasets

#note this file is used for training a model
#for inference and deployment, see other files
import os
from IPython import display as dp
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image
from roboflow import Roboflow


#current working directory
CWD = os.getcwd()

#health checks
dp.clear_output()
ultralytics.checks() #this steps takes about a minute

os.chdir("datasets")
rf = Roboflow(api_key = "glBugZY3xZpsMKjTkGF2") #api key here is generated from my personal roboflow account
project = rf.workspace("machine-learning-hocjz").project("672-yolo")
version = project.version(4) #version of the dataset
dataset = version.download("yolov8")
dataloc = os.path.join(dataset.location, "data.yaml")
os.chdir(CWD)

model = YOLO("yolov8m.pt")
results = model.train(data = dataloc, epochs = 50, imgsz = 640, device = "cpu") #torch doesnt work with cuda 12.3 right now, if you have a compatible version of cuda, device can be changed to gpu

#run the following command for prediction
#"images" can be replaced with whatever the folder is named where all the images are stored
#best.pt is the name of the model file used
#results are saved under /runs/detect/predict/*
#command below
#yolo predict model=best.pt source=images imgsz=640