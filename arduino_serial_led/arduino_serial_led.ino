// Companion for alpha_paradigm.py (dynamic dimming).
//
// Wiring for dimming (external LED):
//   Arduino ~9 -> resistor -> LED long leg
//   LED short leg -> GND
//
// Serial protocol (115200 baud):
//   L<0..255>\n  sets PWM brightness on pin 9
//   1 / 0         still toggles onboard LED on pin 13 (optional)

const int ONBOARD_LED_PIN = 13;
const int PWM_LED_PIN = 9;
String lineBuf = "";

void setup() {
  Serial.begin(115200);
  pinMode(ONBOARD_LED_PIN, OUTPUT);
  digitalWrite(ONBOARD_LED_PIN, LOW);
  pinMode(PWM_LED_PIN, OUTPUT);
  analogWrite(PWM_LED_PIN, 0);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '1') {
      digitalWrite(ONBOARD_LED_PIN, HIGH);
      continue;
    }
    if (c == '0') {
      digitalWrite(ONBOARD_LED_PIN, LOW);
      continue;
    }
    if (c == '\n' || c == '\r') {
      if (lineBuf.length() > 0) {
        if (lineBuf.charAt(0) == 'L') {
          int val = lineBuf.substring(1).toInt();
          val = constrain(val, 0, 255);
          analogWrite(PWM_LED_PIN, val);
        }
        lineBuf = "";
      }
    } else {
      lineBuf += c;
      if (lineBuf.length() > 32) {
        lineBuf = "";
      }
    }
  }
}
