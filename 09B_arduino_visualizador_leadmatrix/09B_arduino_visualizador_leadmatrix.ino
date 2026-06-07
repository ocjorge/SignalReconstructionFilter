#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

// ✔
const uint32_t ICON_OK[3] = {
  0x00000000,
  0x00081020,
  0x40000000
};

// ✖
const uint32_t ICON_REJECT[3] = {
  0x20088140,
  0x01402208,
  0x80000000
};

// RESTORE animación
const uint32_t ICON_RESTORE_1[3] = {
  0x0007C444,
  0x44447C00,
  0x00000000
};

const uint32_t ICON_RESTORE_2[3] = {
  0x0007C6C4,
  0x46C47C00,
  0x00000000
};

const uint32_t ICON_RESTORE_3[3] = {
  0x0007ECE4,
  0x4EC47C00,
  0x00000000
};

String getCommand(String msg) {
  int commaPos = msg.indexOf(',');
  if (commaPos == -1) return msg;
  return msg.substring(0, commaPos);
}

void showOK() {
  matrix.loadFrame(ICON_OK);
}

void showReject() {
  matrix.loadFrame(ICON_REJECT);
}

void showRestoreStatic() {
  matrix.loadFrame(ICON_RESTORE_1);
}

void animateRestore() {
  matrix.loadFrame(ICON_RESTORE_1);
  delay(90);
  matrix.loadFrame(ICON_RESTORE_2);
  delay(90);
  matrix.loadFrame(ICON_RESTORE_3);
  delay(90);
  matrix.loadFrame(ICON_RESTORE_2);
  delay(90);
  matrix.loadFrame(ICON_RESTORE_1);
}

void setup() {
  Serial.begin(115200);
  matrix.begin();
  showReject();
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    if (msg.length() == 0) return;

    String cmd = getCommand(msg);

    if (cmd == "OK") {
      showOK();
      Serial.println("ACK_OK");
    }
    else if (cmd == "REJECT") {
      showReject();
      Serial.println("ACK_REJECT");
    }
    else if (cmd == "RESTORE") {
      animateRestore();
      showRestoreStatic();
      Serial.println("ACK_RESTORE");
    }
    else {
      showReject();
      Serial.println("ACK_UNKNOWN");
    }
  }
}