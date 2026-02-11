/*************************************************
 Project : AI-Based Smart Irrigation System
 Author  : Anusha Ilyas
 Board   : Arduino UNO R4 WiFi
 Version : AI + Serial Integrated v5.0
*************************************************/

#define BLYNK_PRINT Serial

// ---------------- BLYNK ----------------
#define BLYNK_TEMPLATE_ID "TMPL6MaMtQ48N"
#define BLYNK_TEMPLATE_NAME "Smart Farming System"
#define BLYNK_AUTH_TOKEN "1OO0SjDXfCVIztrxUpdzkHxbnuDyrwJr"

// ---------------- LIBRARIES ----------------
#include <SPI.h>
#include <WiFiS3.h>
#include <BlynkSimpleWifi.h>
#include <DHT.h>
#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>

// ---------------- WIFI ----------------
char ssid[] = "Redmi 12C";
char pass[] = "12345678A";

// ---------------- LCD ----------------
hd44780_I2Cexp lcd;

// ---------------- PINS ----------------
#define SOIL_PIN   A0
#define RELAY_PIN  4
#define LED_PIN    7
#define DHTPIN     A3
#define DHTTYPE    DHT11

// ---------------- OBJECTS ----------------
DHT dht(DHTPIN, DHTTYPE);
BlynkTimer timer;

// ---------------- VARIABLES ----------------
int soilValue = 0;
float temperature = 0;
float humidity = 0;

bool pumpState = false;
bool manualMode = false;
bool manualPump = false;
int aiDecision = 0;   // 0=OFF, 1=ON

// ---------------- APPLY PUMP STATE ----------------
void applyPump(bool on) {
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
  digitalWrite(LED_PIN, on ? HIGH : LOW);
  pumpState = on;
}

// ---------------- SENSOR TASK ----------------
void sendSensor() {
  soilValue = analogRead(SOIL_PIN);
  temperature = dht.readTemperature();
  humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT read failed!");
    return;
  }

  // ----------- CONTROL LOGIC -----------
  if (manualMode) {
    applyPump(manualPump);
  } else {
    applyPump(aiDecision == 1);
  }

  // ----------- LCD -----------
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(temperature,1);
  lcd.print("C H:");
  lcd.print(humidity,0);
  lcd.print("%");

  lcd.setCursor(0, 1);
  lcd.print("Soil:");
  lcd.print(soilValue);
  lcd.print(" ");
  lcd.print(pumpState ? "ON" : "OFF");

  // ----------- SERIAL FORMAT (PYTHON COMPATIBLE) -----------
  Serial.print("Soil=");
  Serial.print(soilValue);
  Serial.print(" | Temp=");
  Serial.print(temperature,1);
  Serial.print("C | Hum=");
  Serial.print(humidity,1);
  Serial.print("% | Pump=");
  Serial.print(pumpState ? "ON" : "OFF");
  Serial.print(" | Mode=");
  Serial.println(manualMode ? "Manual" : "Auto");

  // ----------- BLYNK -----------
  Blynk.virtualWrite(V0, temperature);
  Blynk.virtualWrite(V1, humidity);
  Blynk.virtualWrite(V2, soilValue);
  Blynk.virtualWrite(V3, pumpState ? 255 : 0);
  Blynk.virtualWrite(V4, manualMode ? 255 : 0);
}

// ---------------- SERIAL AI COMMAND ----------------
void readAICommand() {
  if (Serial.available() > 0 && !manualMode) {
    char c = Serial.read();
    if (c == '0' || c == '1') {
      aiDecision = c - '0';
    }
  }
}

// ---------------- BLYNK CALLBACKS ----------------
BLYNK_WRITE(V5) {   // Manual Mode
  manualMode = param.asInt();
}

BLYNK_WRITE(V6) {   // Manual Pump
  manualPump = param.asInt();
}

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(9600);

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  applyPump(false);

  dht.begin();

  lcd.begin(16,2);
  lcd.backlight();
  lcd.print("Smart Farming");

  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) delay(500);

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  timer.setInterval(1000L, sendSensor);

  Serial.println("AI Smart Irrigation Started");
}

// ---------------- LOOP ----------------
void loop() {
  Blynk.run();
  timer.run();
  readAICommand();   // 👈 VERY IMPORTANT
}
