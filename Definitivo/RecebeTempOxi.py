import serial
import time

c = 0
strdata = ''


while c < 100000:
    strdata = ''
    arduinoData = serial.Serial('COM3', 115200, timeout=.1)

    while strdata != 't':
        arduinoData.write(bytes('t', 'utf-8'))
        #print("t enviado")
        time.sleep(0.5)
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        print(strdata)
        time.sleep(0.5)
    #print('Saiu do envio de t')
    while strdata != 'f':
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        time.sleep(0.5)
        print(strdata)
    #print('saiu da leitura de t')
    time.sleep(0.5)
    while strdata != 'o':
        arduinoData.write(bytes('o', 'utf-8'))
        time.sleep(0.5)
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        print(strdata)
        time.sleep(0.5)
    #print('saiu do envio de o')
    while (strdata != 'f'):
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        print(strdata)
        time.sleep(1)
    arduinoData.close()
    #print('saiu da leitura de o')

