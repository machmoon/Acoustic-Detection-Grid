#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <PubSubClient.h>
#include <LittleFS.h>
#include <String.h>

#define UART_TX D6
#define UART_RX D7

// WiFi related parameters and setups
const char* ssid     = "Vamsi_Phone";
const char* password = "qwertyuiop";

const char* mqtt_server = "test.mosquitto.org";

const uint16_t mqtt_port = 443;

WiFiClient espClient;
PubSubClient client(espClient);

File logFile;

void serialEvent1();
void initWiFi();
void initUART();

void setup() {
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(250000);
  delay(10);

  initUART();
  //Delete it in the final:
  delay(5000);

  if (!LittleFS.begin(true)) {   // true = format if mount fails
    Serial.println("LittleFS mount failed");
    while (1);
  }

  logFile = LittleFS.open("../audio.txt", FILE_APPEND);
  if (!logFile) {
    Serial.println("Failed to open file");
    while (1);
  }
  delay(5000);
  //initWiFi();

}

void loop() {
  /*
  Serial.println("Test");
  if (!client.connected()) {
    // Reconnect to MQTT broker
    if (client.connect("ESP32Client")) {
      Serial.println("Connected to MQTT broker");
      // Subscribe to MQTT topic after reconnection
      client.subscribe("goontronics/esp32");
    }
  }
  
  client.loop();

  client.publish("goontronics/esp32", "Hello from ESP32");
  delay(2000);
  */

  while (Serial1.available()) {
    char c = Serial1.read();
    logFile.println(c);
  }
}

void serialEvent1() {
  while(Serial1.available()) {
    uint8_t incomingByte = Serial1.read();
  }
}

void initUART() {
  Serial1.begin(10000000, SERIAL_8N1, UART_RX, UART_TX);
  attachInterrupt(digitalPinToInterrupt(UART_RX), serialEvent1, FALLING);
}

void initWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi ..");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }

  Serial.println();

  Serial.print("WiFi status: "); Serial.println(WiFi.status());
  Serial.print("IP: "); Serial.println(WiFi.localIP());
  Serial.print("Gateway: "); Serial.println(WiFi.gatewayIP());
  Serial.print("DNS: "); Serial.println(WiFi.dnsIP());

    // Set the MQTT broker server IP address and port
  client.setServer(mqtt_server, mqtt_port);

  

  // Connect to MQTT broker
  while (!client.connected()) {
    if (client.connect("ESP32Client")) {
      Serial.println("Connected to MQTT broker");
    } else {
      Serial.print("Failed to connect to MQTT broker, rc=");
      Serial.println(client.state());
      delay(1000);
    }
  }

  // Subscribe to MQTT topic
  client.subscribe("goontronics/esp32");
}
