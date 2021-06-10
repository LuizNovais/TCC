#include <SPI.h>
#include <Wire.h>
#include <SparkFunMLX90614.h>     // Biblioeca do Sensor Infravermelho
#include "MAX30100_PulseOximeter.h" // Biblioteca Oximetro

#define REPORTING_PERIOD_MS     1000

IRTherm temperatura; // Sensor de temperatura

PulseOximeter pox; // Sensor de oxigenação
uint32_t tsLastReport = 0;
char carac; 
int count = 0;

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(1);
    pinMode(7,INPUT); 
}

void setupterm(){
    temperatura.begin(0x5A);                                      // Inicia o Sensor no endereço 0x5A
    temperatura.setUnit(TEMP_C);                                  // Define a temperatura em Celsius
}

void setupoxi(){
    count = 0;
    pox.begin();
    if (!pox.begin()) {
        Serial.println("FAILED"); //Testa a comunicação com o oximetro.
        for(;;);
    } 
}

void loop() {
    delay(100);                                                   // Aguarda 0,1 segundos
    if (Serial.available() > 0){                                  // Se a comunicação serial estiver disponível
        carac = Serial.read();                                      // Realiza leitura do barramento serial
        if (carac == 'p'){
            if (digitalRead(7)){
                Serial.println('p'); 
            } 
            if (digitalRead(7)==LOW){
                Serial.println('f');      
            }
        }
        if (carac == 't'){                                          // Se o o caractere lido for t
            setupterm();                                                // Inicia o setup do sensor de temperatura
            Serial.println(carac);                                      // Envia o caracter recebido 
            delay(100);
            while (count < 10){                                         // Loop para enviar 10 amostras 
                if (temperatura.read())  {                              // Solicita a leitura de temperatura ao termômetro
                  Serial.println(temperatura.object());                 // Envia a temperatura lida em graus Celsius
                  delay(100);                                           // Aguarda 0,1 segundos
                  count++;                                              // Incremento da variável iterativa
                }
            }
            Serial.println('f');                                        // Envia o caractere f para informar que finalizou o envio.
            Serial.flush();                                             // Função que garante que a informação se mantenha no barramento até ser enviada completamente.
            delay(1000);                                                // Aguarda 1 segundo
            carac = 'n';                                                // Seta a variável com outro caractere
            }
        }
        if (carac == 'o'){                                              // Se o o caractere lido for o
            setupoxi();                                                 // Inicia o setup do sensor de oxigenação
            pox.update();
            Serial.println(carac); 
            while (count < 10){                                         // Loop para enviar 10 amostras
                if (millis() - tsLastReport > REPORTING_PERIOD_MS) {    // Se o tempo que passou desde o início de execução menos o tempo do último envio for maior que 1 segundo
                    Serial.println(pox.getSpO2());                      // Envia a oxigenação em porcentagem
                    count++;                                            // Incremento da variável iterativa
                    tsLastReport = millis();                            // Define o tempo do último envio, como o tempo atual.
                }
                pox.update();                                           // Solicita nova leitura da oxigenação.                                    
            }
            pox.update();
            count = 0;                                                      // Zera a variável de iteração.
            carac = 'n';                                                    // Seta a variável com outro caractere
            Serial.println('f');                                            // Envia o caractere f para informar que finalizou o envio.
        }
    }
