import serial
import time


def enviainfo():
    c = 0
    strdata = ''
    temps = []
    oxis = []
    strdata = ''
    arduinoData = serial.Serial('COM3', 115200, timeout=.1)

    while strdata != 't':
        arduinoData.write(bytes('t', 'utf-8'))
        #print("t enviado")
        time.sleep(0.5)
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        #print(strdata)
        time.sleep(0.5)
    #print('Saiu do envio de t')
    while strdata != 'f':
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        time.sleep(0.1)
        if strdata != 'f': temps.append(strdata)
        #print(strdata)
    #print('saiu da leitura de t')
    time.sleep(0.5)
    while strdata != 'o':
        arduinoData.write(bytes('o', 'utf-8'))
        time.sleep(0.5)
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        #print(strdata)
        time.sleep(0.5)
    #print('saiu do envio de o')
    while (strdata != 'f'):
        data = arduinoData.readline()
        strdata = data.rstrip().decode('utf-8')
        if strdata != 'f': oxis.append(strdata)
        #print(strdata)
        time.sleep(1)
    j = 0
    arduinoData.close()
    return(temps, oxis)

print(enviainfo())
