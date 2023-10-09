import cv2
import sys
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

#%config InlineBackend.figure_format = 'svg'
'''program_name = sys.argv[0]
arguments = sys.argv[1:]
co = len(arguments)
'''

td=0.51

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.51 ,
    'gpu': 0.7
}
tfnet = TFNet(options)
cnt=0

img = cv2.imread('qwe.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# use YOLO to predict the image
result = tfnet.return_predict(img)
for res in result:
	if res['label']=='person':
	    print(res['confidence'])
	    cnt=cnt+1
print('Count in the image:',cnt)

# read the color image and covert to RGB
'''for itr in range(1,17):
	src = str(itr)+'.jpg';
	img = cv2.imread(src, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# use YOLO to predict the image
	result = tfnet.return_predict(img)
	cnt=1
	img.shape
	for res in result:
	    if res['label']=='person':
	        print(res['confidence'])
	        cnt=cnt+1
	print('Count in the image:', itr ,cnt)'''

'''if cv2.waitKey(0):
cv2.imshow('img',img)
	exit(0)'''



