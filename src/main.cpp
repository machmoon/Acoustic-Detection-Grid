#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <PubSubClient.h>

#define UART_TX D6
#define UART_RX D7

#define UART_BUF_LENGTH 30000

// WiFi credentials
const char* ssid = "OwenLiu66";
const char* password = "Owen451073*";

// MQTT config
const char* mqtt_server = "test.mosquitto.org";
const char* mqtt_topic = "goontronics/esp32";
const uint16_t mqtt_port = 1883;

// Audio buffers
uint8_t uartBufferMic_1[UART_BUF_LENGTH];
uint8_t uartBufferMic_2[UART_BUF_LENGTH];
uint8_t uartBufferMic_3[UART_BUF_LENGTH];

WiFiClient espClient;
PubSubClient client(espClient);

// Function declarations
void checkForAudio();
void initWiFi();
void initUART();
bool ensureMqttConnected();

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(250000);
  
  initUART();
  
  delay(3000);  // Give time to open serial monitor
  Serial.println("ESP32 Audio Node Starting...");
  
  initWiFi();
  
  // CRITICAL: Increase MQTT buffer size for large payloads (default is only 256 bytes!)
  client.setBufferSize(UART_BUF_LENGTH + 200);
  Serial.print("MQTT buffer size set to: ");
  Serial.println(UART_BUF_LENGTH + 200);
}

void loop() {
  // Keep MQTT connection alive
  client.loop();
  
  // Ensure we're connected before doing anything
  if (!ensureMqttConnected()) {
    delay(500);
    return;
  }
  
  // Check for incoming audio data
  checkForAudio();
}

void checkForAudio() {
  while (Serial1.available()) {
    uint8_t incomingByte = Serial1.read();
    
    // Trigger condition: byte outside normal range indicates start of audio frame
    if (incomingByte > 185) {
      digitalWrite(LED_BUILTIN, HIGH);  // LED on while recording
      Serial.println("🎤 Trigger detected! Reading audio into buffer...");
      
      // Fill the buffer from UART
      for (uint32_t i = 0; i < UART_BUF_LENGTH; i++) {
        while (!Serial1.available()) {
          // Wait for data (could add timeout here)
        }
        uartBufferMic_1[i] = Serial1.read();
      }
      
      Serial.print("✅ Buffer full: ");
      Serial.print(UART_BUF_LENGTH);
      Serial.println(" bytes");
      
      digitalWrite(LED_BUILTIN, LOW);  // LED off
      
      // Ensure MQTT is connected before publishing
      if (!ensureMqttConnected()) {
        Serial.println("❌ MQTT not connected, dropping frame");
        return;
      }
      
      // Send frame start indicator
      Serial.println("📤 Sending frame start signal...");
      bool startSent = client.publish(mqtt_topic, "Goontronics Engaged");
      if (!startSent) {
        Serial.println("❌ Failed to send frame start");
        return;
      }
      
      client.loop();  // Process any pending MQTT work
      delay(50);      // Small delay to ensure frame start is received first
      
      // Send entire audio buffer in ONE message
      Serial.println("📤 Publishing audio buffer...");
      unsigned long startTime = millis();

      bool success = client.publish(mqtt_topic, uartBufferMic_1, UART_BUF_LENGTH);
      
      unsigned long elapsed = millis() - startTime;
      
      if (success) {
        Serial.print("✅ Published ");
        Serial.print(UART_BUF_LENGTH);
        Serial.print(" bytes in ");
        Serial.print(elapsed);
        Serial.println("ms");
      } else {
        Serial.println("❌ Publish failed!");
        Serial.println("   Check: Is buffer size set? Is MQTT connected?");
      }
      
      client.loop();  // Process any pending MQTT work
    }
  }
}

void initUART() {
  Serial1.begin(2500000, SERIAL_8N1, UART_RX, UART_TX);
  Serial.println("UART1 initialized at 2.5Mbaud");
}

void initWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  
  Serial.println();
  Serial.println("✅ WiFi connected!");
  Serial.print("   IP: ");
  Serial.println(WiFi.localIP());
  
  // Setup MQTT server
  client.setServer(mqtt_server, mqtt_port);
  Serial.print("   MQTT server: ");
  Serial.println(mqtt_server);
}

bool ensureMqttConnected() {
  // If already connected, return immediately
  if (client.connected()) {
    return true;
  }
  
  // Not connected - try to connect
  Serial.print("MQTT connecting...");
  
  // Use unique client ID based on MAC address
  String clientId = "esp32-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  
  if (client.connect(clientId.c_str())) {
    Serial.println(" ✅ connected!");
    return true;
  }
  
  Serial.print(" ❌ failed, rc=");
  Serial.println(client.state());
  /*
    MQTT state codes:
    -4 : MQTT_CONNECTION_TIMEOUT
    -3 : MQTT_CONNECTION_LOST
    -2 : MQTT_CONNECT_FAILED
    -1 : MQTT_DISCONNECTED
     0 : MQTT_CONNECTED
     1 : MQTT_CONNECT_BAD_PROTOCOL
     2 : MQTT_CONNECT_BAD_CLIENT_ID
     3 : MQTT_CONNECT_UNAVAILABLE
     4 : MQTT_CONNECT_BAD_CREDENTIALS
     5 : MQTT_CONNECT_UNAUTHORIZED
  */
  return false;
}