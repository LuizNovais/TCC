#include <SPI.h>
#include <Wire.h>
#include <SparkFunMLX90614.h>     // Biblioeca do Sensor Infravermelho
#include "MAX30100_PulseOximeter.h" // Biblioteca Oximetro

#define REPORTING_PERIOD_MS     1000 //oxi

IRTherm temperatura; //temp

PulseOximeter pox; //oxi
uint32_t tsLastReport = 0; //oxi
char carac;
int count = 0;

void onBeatDetected()
{
    Serial.println("Beat!");
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
}

void setupterm(){
  temperatura.begin(0x5A);         // Inicia o Sensor no endereço 0x5A
  temperatura.setUnit(TEMP_C);     // Define a temperatura em Celsius
  //Serial.print("Termometro inicializado\n");
}

void setupoxi(){
 pox.begin();
  if (!pox.begin()) {
        Serial.println("FAILED");
        for(;;);
  } else {
      //Serial.println("SUCCESS\n");
  }
 // pox.setOnBeatDetectedCallback(onBeatDetected);
  //Serial.print("Oximetro inicilizado\n");
}

void loop() {
  delay(100);
  if (Serial.available() > 0){
    carac = Serial.read();  
    if (carac == 't'){
      setupterm();
      Serial.println(carac);
      if (temperatura.read())  {
      //Serial.println("Objeto: ");
      Serial.println(temperatura.object());
      //Serial.println("Ambiente: ");
      //Serial.println(temperatura.ambient());
      Serial.println('f');
      Serial.flush();
      delay(1000);                   // Aguarda 2 segundos
      carac = 'n';
      }
    }
    if (carac == 'o'){
      setupoxi(); 
      Serial.println(carac); 
      // Asynchronously dump heart rate and oxidation levels to the serial
      // For both, a value of 0 means "invalid"
      while (count < 10){
        if (millis() - tsLastReport > REPORTING_PERIOD_MS) {
           // Serial.println("Heart rate:");
           // Serial.println(pox.getHeartRate());
           // Serial.println("SpO2:");
            Serial.println(pox.getSpO2());
            //Serial.flush();
            count++;
            tsLastReport = millis();
       //     Serial.println(values[count]);  
        }
            pox.update();
      }
      
    count = 0;
    carac = 'n';
    Serial.println('f');
    
    }
  }
}
