# Python script to calculate path loss:
#
# Syntax:
#           python3 ./simrssi.py input_file.csv output_file.csv
#
#	modify source_x and source_y to the decimal degree values of the lat/lon of the RF source
#
#	the input file should be a comma separated value (CSV) resembling the format below:
#	Note: 	the first row is the header labels
#		date is YYYYMMDD
#		time is UTC seconds
#		remrssi is remote RSSI as measured by the remote device or drone (no units)
#		rssi is the RSSI measured at the source (no units)
#		noise is the noise measured at the source (no units)
#		remnoise is the noise measured at the remote device (no units)
#		lat/lon are position in decimal degrees
#
# date,time,rssi,remrssi,noise,remnoise,lat,lon
# 20210227,163542,154,150,27,44,39.034535,-76.655579
# 20210227,163542,154,150,27,44,39.034534,-76.655579
#
#	the output file will be a comma separated value (CSV) resembling the format below:
#	Note: 	the first row is the header labels
#		remrssi is remote RSSI as measured by the remote device or drone (no units)
#		rssi is the RSSI measured at the source (no units)
#		simrssi is the calculated rssi at the lat/lon from the input file (no units)
#		lat/lon are position in decimal degrees
#
# rssi,remrssi,simrssi,lat,lon
# 154,150,126,39.034535,-76.655579
import csv
import math
import os
import sys

import matplotlib.cm as cmap
import matplotlib.pyplot as mpl
import numpy as np

freq = 915000000  # Hz
c = 300000000*3.28  # ft/sec
lamda = c/freq

# lat/lon in decimal degrees of RF source
source_x, source_y = 39.034554, -76.655594

# Variable declaration
# hb=100; #in feets(height of BS antenna)
# hm=5;  # in feets(height of mobile antenna)
# f=881.52;#in MHz
# lamda=1.116;  #in feet
# d=5000;  #in feetpython3 ./simrssi.py Test6.csv out6.csv

Gb = 10**0.2   # 2dB(BS antenna gain)
Gm = 10**0.1   # 1dB (Mobile antenna gain)
Tx = 15         # in dBm
Sigma = 2     # Noise power in dB
# No units (This is used to correct for sensor range which can vary between manufacturers)
Offset = 148

# Function to calculate free space path loss from lat/lon
#


def get_rssi(tx_power, source_x, source_y, current_x, current_y):

    # calculate the distance from lat/lon coords in feet
    #
    # Note: one degree of latitude = approx. 364000 feet, one degree of longitude = approx. 288200 feet
    #

    d1 = math.sqrt((((current_x-source_x)*364000)**2) +
                   (((current_y-source_y)*288200)**2))
    d = round(d1, 2)

    # add guassian noise
    noise = np.random.normal(0, Sigma)

    # free space attenuation
    #
    # Using the Friis transmission equation, see: https://en.wikipedia.org/wiki/Free-space_path_loss

    free_atten = (4*math.pi*d/lamda)**2*(Gb*Gm)**-1

    y = (10*math.log10(free_atten+.0001))
    print('Free space attenuation is %d dB' % y)
    print('Distance is %.2f feet' % d)
    return round(tx_power-y+noise+Offset)


def main():
    file_in_path = sys.argv[1]		# input file
    file_out_path = sys.argv[2]		# output file

    if not os.path.isfile(file_in_path):
        print('File path {} does not exist. Exiting...'.format(file_in_path))
        sys.exit()

    out_file = open(file_out_path, 'w')
    out_writer = csv.writer(out_file, delimiter=',')

    with open(file_in_path) as in_file:
        in_reader = csv.reader(in_file, delimiter=',')
        line_count = 0
        for row in in_reader:
            if line_count == 0:
                print(f"Column names are {', '.join(row)}")
                out_writer.writerow(
                    ['rssi', 'remrssi', 'simrssi', 'lat', 'lon'])
                # out_writer.writerow([row[0],row[1],row[2]])
                #print('Field names are:' + ', '.join(field for field in fields))
                line_count += 1
            else:
                #print ('row4 %s '%row[4])
                simrssi = get_rssi(Tx, source_x, source_y,
                                   float(row[6]), float(row[7]))
                out_writer.writerow([row[2], row[3], simrssi, row[6], row[7]])
                # print(f"\t{row[0]},{row[1]},{row[2]}")  # older format without date/timestamp

                line_count += 1

    print(f'Processed {line_count} lines.')


if __name__ == '__main__':
    main()
