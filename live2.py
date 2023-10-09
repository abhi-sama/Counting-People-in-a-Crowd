import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.51,
    'gpu': 0.8
}
frame_height=1280
frame_width=720
tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width )
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

cnt=0
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        cnt=0
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            #print(tl,'\t',br,'\t',label,'\n')
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)   
            if label='person':
                cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

'''if(result['label']=='person'):
                personCoordinates.append(result)
            if label=='person':    
                for var1 in personCoordinates:
                    cnt=cnt+1'''