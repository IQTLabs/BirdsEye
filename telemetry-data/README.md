
## BirdsEye Telemetry

Drone controllers use a variety of protocols and modulation schemes to control and communicate with the drone vehicle. Some of the most popular come from DJI and they are called Lightbridge, Lightbridge-2, Ocusync and Ocusync-2. These protocols use a combination of OFDM and FHSS modulation techniques to receive data and send control signals to the drone. 

Building radios to receive OFDM and FHSS signals is not trivial, even using software defined radio(SDR) technology can still be a substantial effort to implement these receivers. For this project, development of the receiver is out of scope. Instead, we are using telemetry radios that are commonly found on commercial drones to send and receive signal strength data. 

A next step for this project would be to design and implement the actual SDR based receivers needed to decode these signals. However, for now, we are mainly focused on developing algorithms for geolocating the drone controller. 

The table below shows protocols and other information for various drone manufacturers.

**Make**|**Model**|**Freq (MHz)**|**TX Power (dBm)**|**Protocol**|**Comments**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
DJI|Phantom 3|5725 - 5825|+19|Lightbridge| 
NXP|Telemetry Controller|433/915|+20|FHSS|HGD-TELEM433
NXP|Fly Sky|2405.5-2475|<20|GFSK/AFHDS 2A|FS-i6S
DJI|Phantom 4 Pro|2400-2483 /5725-5825|+26/+28|Lightbridge| 
DJI|Phantom 4 Pro v2.0|2400-2483 /5725-5850|+26/+26|OcuSync| 
DJI|Mavic Pro|2400-2483|+26|OcuSync| 
DJI|Inspire 2|2400-2483 /5725-5850|+26/+28|Lightbridge-2| 
DJI |Matrice 200|2400-2483 /5725-5850|+26/+26|Lightbridge-2| 
DJI|Matrice 600 Pro|2400-2483 /5725-5825|+20/+13|Lightbridge-2| 

### Mavlink

MAVLink or Micro Air Vehicle Message Marshalling Library is a very lightweight, header-only message library for communication between drones and/or ground control stations and between onboard drone components. It consists primarily of message-set specifications for different systems.

MAVLink follows a modern hybrid publish-subscribe and point-to-point design pattern: Data streams are sent / published as topics while configuration sub-protocols such as the mission protocol or parameter protocol are point-to-point with retransmission.

Messages are defined within XML files. Each XML file defines the message set supported by a particular MAVLink system, also referred to as a "dialect". The reference message set that is implemented by most ground control stations and autopilots is defined in common.xml (most dialects build on top of this definition).

The MAVLink toolchain uses the XML message definitions to generate MAVLink libraries for each of the supported programming languages. Drones, ground control stations, and other MAVLink systems use the generated libraries to communicate. MAVLink was used to transport signal strength data for this project.

### QGroundControl

QGroundControl provides full flight control and vehicle setup for PX4 or ArduPilot powered vehicles. It provides easy and straightforward usage for beginners, while still delivering high end feature support for experienced users. QGroundControl was used to collect the data used for this project. 
