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
def getinfo():								# Função que abre comunicação serial com o microcontrolador e recebe as medições
	c = 0								# Variável iterativa
	str_data = ''							# String que recebe as medições uma a uma na leitura da porta serial
	temp_l = []							# Lista que recebe as 10 medições de temperatura.
	oxi_l = []							# Lista que recebe as 10 medições de oxigenação.
	arduino = serial.Serial('COM3', 115200, timeout=.1)		# Variável que inicia a comunicação serial com o microcontrolador, como parâmetros:
									# 'COM3': Porta USB que o microcontrolador está conectado; 115200: Baud Rate
	while str_data != 't':						# Loop que envia o caracter t ao microcontrolador solicitando as medições de temperatura
		arduino.write(bytes('t', 'utf-8'))			# Escreve o caractere t na porta serial
		time.sleep(0.5)						# Aguarda 0,5 segundos
		data = arduino.readline()				# Lê o conteúdo da porta serial e aloca na variável data
		str_data = data.rstrip().decode('utf-8')		# Decodifica o dado hexadecimal e aloca na variável str_data 
		time.sleep(0.5)						# O programa fica repetindo o envio de t, até receber do microcontrolador um t como confirmação de que está pronto para o envio das medições
	while str_data != 'f':						# Loop para leitura e arquivo das medições, até que receba o caractere f que sinaliza o fim do envio de medições
		data = arduino.readline()				# Lê o conteúdo da porta serial e aloca na variável data
		str_data = data.rstrip().decode('utf-8')		# Decodifica o dado hexadecimal e aloca na variável str_data 
		if str_data != 'f': temp_l.append(str_data)		# Guarda o valor recebido em strdata como um novo item na lista temp_l
		time.sleep(0.1)						# Aguarda 0,1 segundos
	time.sleep(0.5)							# Aguarda 0,5 segundos
	while str_data != 'o':						# Loop que envia o caracter o ao microcontrolador solicitando as medições de oxigenação
		arduino.write(bytes('o', 'utf-8'))			# Escreve o caractere o na porta serial
		time.sleep(0.5)						# Aguarda 0,5 segundos
		data = arduino.readline()				# Lê o conteúdo da porta serial e aloca na variável data
		str_data = data.rstrip().decode('utf-8')		# Decodifica o dado hexadecimal e aloca na variável str_data 
		time.sleep(0.5)						# O programa fica repetindo o envio de o, até receber do microcontrolador um o como confirmação de que está pronto para o envio das medições
	while (str_data != 'f'):					# Loop para leitura e arquivo das medições, até que receba o caractere f que sinaliza o fim do envio de medições
		data = arduino.readline()				# Lê o conteúdo da porta serial e aloca na variável data
		str_data = data.rstrip().decode('utf-8')		# Decodifica o dado hexadecimal e aloca na variável str_data 
		if str_data != 'f': oxi_l.append(str_data)		# Guarda o valor recebido em strdata como um novo item na lista temp_l
		time.sleep(1)						# Aguarda 1 segundos, o oximetro necessita de um tempo maior para atualizar as medições
	arduino.close()							# Encerra a comunicação serial com o microcontrolador
	return(temp_l, oxis)						# Retorna as duas listas como resultado da função
def oxige():								# Função que trata as medições recebidas e prepara para exibição
	global ox1							# Variável que recebe o valor de oxigenação a ser exibido 
	global ox2							# Lista de medições confiáveis de oxigenação
	global temp1							# Variável que recebe o valor de temperatura a ser exibido
	global temp2							# Lista de medições confiáveis de oxigenação
	oxix=[]								# Variável local que recebe a lista de medições de oxigenação
	tempx=[]							# Variável local que recebe a lista de medições de temperatura
	double_list=getinfo()						# Chama a função getinfo() e aloca as duas listas com as medições na variáve double_list
	oxix.append(double_list[1])					# Aloca a lista de medições de oxigenação na oxix
	ox = oxix[0]							# Formatação da lista
	for i in ox:							
		if float(i) > 10 and float(i) < 101:			# Iteração para transformar os valores texto em numéricos e para ignorar valores abaixo de 10% e acima de 101%
			ox2.append(i)
	ox1 = str(statistics.mode(ox2))					# Dos valores que estão na faixa selecionada, realiza a moda e aloca na variável ox1
	tempx.append(double_list[0])					# Aloca a lista de medições de temperatura na tempx
	temp = tempx[0]							# Formatação da lista
	for i in temp:
		if float(i) > 10 and float(i) < 101:			# Iteração para transformar os valores texto em numéricos e para ignorar valores abaixo de 10% e acima de 101%
			temp2.append(i)
	temp1 = str(statistics.mode(temp2))				# Dos valores que estão na faixa selecionada, realiza a moda e aloca na variável ox1

# Inicialização das variáveis de controle e construção dos argumentos
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

