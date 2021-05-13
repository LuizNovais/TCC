# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import serial
import time
import threading
import statistics
import concurrent.futures
from plyer import notification

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
def enviainfo():
	c = 0
	strdata = ''
	temps = []
	oxis = []
	strdata = ''
	arduinoData = serial.Serial('COM3', 115200, timeout=.1)

	while strdata != 't':
		arduinoData.write(bytes('t', 'utf-8'))
		print("t enviado")
		time.sleep(0.5)
		data = arduinoData.readline()
		strdata = data.rstrip().decode('utf-8')
		print(strdata)
		time.sleep(0.5)
	print('Saiu do envio de t')
	while strdata != 'f':
		data = arduinoData.readline()
		strdata = data.rstrip().decode('utf-8')
		time.sleep(0.1)
		if strdata != 'f': temps.append(strdata)
		print(strdata)
	print('saiu da leitura de t')
	time.sleep(0.5)
	while strdata != 'o':
		arduinoData.write(bytes('o', 'utf-8'))
		time.sleep(0.5)
		data = arduinoData.readline()
		strdata = data.rstrip().decode('utf-8')
		print(strdata)
		time.sleep(0.5)
		print('saiu do envio de o')
	while (strdata != 'f'):
		data = arduinoData.readline()
		strdata = data.rstrip().decode('utf-8')
		if strdata != 'f': oxis.append(strdata)
		print(strdata)
		time.sleep(1)
	arduinoData.close()
	return(temps, oxis)
def oxige():
	global ox1
	global ox2
	global temp1
	global temp2
	oxix=[]
	tempx=[]
	recebido=enviainfo()
	oxix.append(recebido[1])
	ox = oxix[0]
	for i in ox:
		if float(i) > 10 and float(i) < 101:
			ox2.append(i)
	ox1 = str(statistics.mode(ox2))
	tempx.append(recebido[0])
	temp4 = tempx[0]
	for i in temp4:
		if float(i) > 10 and float(i) < 101:
			temp2.append(i)
	temp1 = str(statistics.mode(temp2))

# construct the argument parser and parse the arguments
ox1="0"
ox2=[]
temp1="0"
temp2=[]
old_ox1="0"
old_temp1="0"
old_label=""
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Com Mascara" if mask > withoutMask else "Sem Mascara"
		color = (0, 255, 0) if label == "Com Mascara" else (0, 0, 255)
		if label == "Sem Mascara":
			label6 = ("Por favor, coloque a mascara")
			org6 = (500, 400)
			cv2.putText(frame, label6, org6,
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		if label != old_label and label == "Com Mascara":
			threading.Thread(target=oxige).start()
			print(ox1)
			print(temp1)
		# include the probability in the label
		if ox1 != old_ox1:
			#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			org = (500,485)
			label1 = (f"Oxigenacao = {ox1}")
			org1 = (500, 515)
			label2 = (f"Temperatura = {temp1}")
			org2 = (500, 545)
			label4 = ("Entrada autorizada")
			org4 = (500, 400)
			label5 = ("Entrada nao autorizada")
			org5 = (500, 400)
			cv2.putText(frame, label, org,
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.putText(frame, label1, org1,
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.putText(frame, label2, org2,
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			print("entrei no if")
			print(float(ox1))
			print(float(temp1))
			if label == "Com Mascara" and float(temp1) < 37.5 and float(ox1) > 89:
				cv2.putText(frame, label4, org4,
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			else:
				cv2.putText(frame, label5, org5,
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			c =+ 1
			if c > 5 or label != old_label:
				old_ox1 = ox1
		else:
			label3 = ("Coloque o dedo no local indicado")
			org3 = (500, 485)
			cv2.putText(frame, label3, org3,
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
		old_label = label
		# display the label and bounding box rectangle on the output
		# frame


	# show the output frame
	#cv2.imshow("Frame", frame)
	#key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

