/*
  Minimal example: print one numeric reading per line for AquaTrack /api/arduino/live.
  Adjust TRIG_PIN, ECHO_PIN, and tank geometry to match your water-level sensor.

  For HC-SR04 ultrasonic: value = distance in cm (lower might mean more water — calibrate in UI/sketch).
*/
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;

long readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL); // timeout ~5m
  if (duration == 0) return -1;
  return (duration * 0.0343) / 2.0;
}

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  long cm = readDistanceCm();
  if (cm >= 0) {
    Serial.print(F("LEVEL:"));
    Serial.println((float)cm, 2);
  } else {
    Serial.println(F("{\"level\":-1}"));
  }
  delay(500);
}
