from ast import arg
import queue
import cv2
import cvzone
from robot_hat import TTS
from multiprocessing import Process
from multiprocessing import Queue
from picarx import Picarx

px = Picarx()

ttsBot = TTS()

def objectDetector(queue):
    thres = 0.55
    nmsThres = 0.2
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().split('\n')
    print(classNames)
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = "frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    
    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(
            img, confThreshold=thres, nmsThreshold=nmsThres)
        try:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cvzone.cornerRect(img, box)
                cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (box[0] + 10, box[1] +
                            30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)
                objectNames = classNames[classId - 1].upper()
                queue.put(objectNames)
                print(objectNames)

        except:
            pass

        cv2.imshow("Image", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

def sayNames(queue):
    
    while True:
        nameOfObject = queue.get()
        ttsBot.say(f"this is {nameOfObject}")


def on_press(key):
    try:
        #print(key.char)
        if key.char == 'w':
            px.set_dir_servo_angle(0)
            px.forward(1)
            
        elif key.char == 's':
            px.set_dir_servo_angle(0)
            px.forward(-1)
            
        elif key.char == 'a':
            px.set_dir_servo_angle(-45)
            px.forward(1)
            
        elif key.char == 'd':
            px.set_dir_servo_angle(45)
            px.forward(1)
            
        elif key.char == 'u':
            #px.set_camera_servo2_angle(0)
            #sleep(1)
            px.set_camera_servo2_angle(45)
        
        elif key.char == 'j':
            #px.set_camera_servo2_angle(0)
            #sleep(1)
            px.set_camera_servo2_angle(-45)
            
        elif key.char == 'h':
            px.set_camera_servo1_angle(-45)
            
        elif key.char == 'k':
            px.set_camera_servo1_angle(45)
            
        else:
            px.set_camera_servo1_angle(0)
            px.set_camera_servo2_angle(0)
            px.set_dir_servo_angle(0)
            px.forward(0)
            
    except AttributeError:
        print('special key {0} pressed'.format(
            key))      

if __name__ == "__main__":
    
    queue = Queue()
    
    p1 = Process(target=objectDetector,args=(queue,))
    p2 = Process(target=sayNames,args=(queue,))
    
    p1.start()
    p2.start()
    
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()