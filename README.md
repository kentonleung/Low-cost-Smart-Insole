# TinyInsoleRecorder
Low-cost wearable gait monitoring system using TinyDuino, force-sensitive resistors (FSRs), and a BMA250 accelerometer.

## Overview
**TinyInsoleRecorder** is firmware for a custom insole-based gait monitoring system.  
It continuously records acceleration and FSR pressure data to an SD card using a TinyDuino microcontroller.  

The device is designed for simple, hands-free operation: **turn it on to start recording** and **turn it off to stop**.

---

## Hardware Requirements
- TinyDuino Processor Board (ATmega328 or compatible)  
- TinyDuino microSD Adapter (chip select pin: 10)  
- BMA250 Accelerometer (I²C connection)  
- Three Force-Sensitive Resistors (FSRs)  
  - FSR L → A0  
  - FSR M → A1  
  - FSR H → A2  
- TinyShield USB Interface (for programming)  
- Battery pack (e.g., 3.7 V Li-ion or Li-Po)  
- Connecting wires or custom insole PCB  

Optional:  
- LED indicator for recording status  

---

## Software Requirements
- [Arduino IDE](https://www.arduino.cc/en/software) (version 2.x recommended)  
- Board package: Arduino AVR Boards  
- Library dependencies:  
  - `Wire.h`  
  - `SPI.h`  
  - `SD.h`  
  - `BMA250.h` (included in this project folder)

---

## Installation and Setup

1. Download or clone this repository.  
2. Open `TinyInsoleRecorder_copy_20250422171109.ino` in Arduino IDE.  
3. Connect the TinyDuino via USB.  
4. In Arduino IDE:  
   - Go to **Tools → Board → Arduino Pro or Pro Mini**  
   - Processor: **ATmega328 (3.3 V, 8 MHz)**  
   - Port: select the correct COM port  
5. Click **Upload**.  

Once uploaded, the device is ready to record.  

---

## How to Use

1. Insert a **formatted microSD card** into the TinyDuino’s SD adapter.  
2. **Power on the device** (via battery or USB). Recording starts automatically.  
3. **Wear or place the insole** during activity.  
4. **Power off the device** to stop recording.  
5. **Remove the SD card** and insert it into a computer to view the data files (for example, `DATA001.TXT`).  

---

## Data Format
Each line in the output file contains:  
```
timestamp, accelX, accelY, accelZ, FSR_L, FSR_M, FSR_H
```

- `timestamp`: milliseconds since device start  
- `accelX, accelY, accelZ`: accelerometer readings from the BMA250  
- `FSR_L, FSR_M, FSR_H`: raw analog readings (0–1023) from the three FSR sensors  

The accelerometer updates approximately every **64 ms**.

---

## Troubleshooting

| Problem | Possible Cause | Solution |
|----------|----------------|-----------|
| No file on SD card | SD card not formatted or loose connection | Format as FAT32 and reinsert |
| No acceleration data | BMA250 not connected or detected | Check wiring or I²C pins |
| FSR values not changing | Loose sensor wiring | Inspect connections at A0–A2 |
| Upload fails | Wrong board or port selected | Recheck Arduino IDE settings |
| Empty file | Device powered off too early | Keep powered during recording |

---

## Notes
- Sampling rate is defined by the BMA250 update setting (`BMA250_update_time_64ms`).  
- Pin mapping and chip select pin can be modified in the `.ino` file.  
- Always insert the SD card before powering on.  
- Use a high-quality SD card (Class 10 recommended).  
- Remove power when not in use to preserve battery life.
