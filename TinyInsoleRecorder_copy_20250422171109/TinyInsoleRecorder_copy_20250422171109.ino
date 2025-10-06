/*************************************************************************
 * BMA250 TinyZero Tutorial:
 * This example program will show the basic method of printing out the 
 * accelerometer values from the BMA250 to the Serial Monitor, and the
 * Serial Plotter
 * 
 * Hardware by: TinyCircuits
 * Code by: Laverena Wienclaw for TinyCircuits
 *
 * Initiated: Mon. 11/1/2018
 * Updated: Tue. 11/1/2018
 ************************************************************************/
 
#include <Wire.h>         // For I2C communication with sensor
#include "BMA250.h"       // For interfacing with the accel. sensor
#include <SPI.h>
#include <SD.h>

// Accelerometer sensor variables for the sensor and its values
BMA250 accel_sensor;
int x, y, z;
double temp;

const int chipSelect = 10;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Wire.begin();

  Serial.print("Initializing BMA...");
  // Set up the BMA250 acccelerometer sensor
  accel_sensor.begin(BMA250_range_2g, BMA250_update_time_64ms); 

    if (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
    // don't do anything more:
    while (1);
  }
  Serial.println("card initialized.");

}

void loop() {
  accel_sensor.read();//This function gets new data from the acccelerometer

  // Get the acceleration values from the sensor and store them into global variables
  // (Makes reading the rest of the program easier)
  int t = millis();
  x = accel_sensor.X;
  y = accel_sensor.Y;
  z = accel_sensor.Z;
  int L = analogRead(A0);
  int M = analogRead(A1);
  int H = analogRead(A2);

  String concatenatedString = String(t) + ", " + String(x) + ", " + String(y) + ", " + String(z) + ", " + String(L) + ", " + String(M) + ", " + String(H);

  SerialUSB.println(concatenatedString);
  // Print the concatenated string
  // If the BME280 is not found, nor connected correctly, these values will be produced
  // by the sensor 
  if (x == -1 && y == -1 && z == -1) {
    // Print error message to Serial Monitor
    Serial.print("ERROR! NO BMA250 DETECTED!");
  }
  
  File dataFile = SD.open("datalog.txt", FILE_WRITE);

  // if the file is available, write to it:
  if (dataFile) {
    dataFile.println(concatenatedString);
    dataFile.close();
    // print to the serial port too:
  }
  // if the file isn't open, pop up an error:
  else {
    Serial.println("error opening datalog.txt");
  }

  // The BMA250 can only poll new sensor values every 64ms, so this delay
  // will ensure that we can continue to read values
  delay(50);
  // ***Without the delay, there would not be any sensor output*** 
}



