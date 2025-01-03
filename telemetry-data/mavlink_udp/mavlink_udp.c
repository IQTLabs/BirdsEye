/*******************************************************************************
 Copyright (C) 2010  Bryan Godbolt godbolt ( a t ) ualberta.ca

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.

 ****************************************************************************/
/*
 This program sends some data to qgroundcontrol using the mavlink protocol.  The sent packets
 cause qgroundcontrol to respond with heartbeats.  Any settings or custom commands sent from
 qgroundcontrol are printed by this program along with the heartbeats.


 I compiled this program sucessfully on Ubuntu 10.04 with the following command


 gcc -std=c99 -I ../../include/common -o mavlink_udp mavlink_udp.c

the rt library is needed for the clock_gettime on linux
 */
/* These headers are for QNX, but should all be standard on unix/linux */
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <time.h>
#if (defined __QNX__) | (defined __QNXNTO__)
/* QNX specific headers */
#include <unix.h>
#else
/* Linux / MacOS POSIX timer headers */
#include <sys/time.h>
#include <time.h>
#include <arpa/inet.h>
#include <stdbool.h> /* required for the definition of bool in C99 */
#endif

/* This assumes you have the mavlink headers on your include path
 or in the same folder as this source file */
#include <mavlink.h>


#define BUFFER_LENGTH 2041 // minimum buffer size that can be used with qnx (I don't know why)

uint64_t microsSinceEpoch();

int main(int argc, char* argv[])
{

	char help[] = "--help";

	struct		tm timer;
	time_t now;
	char buff[100];

	char target_ip[100];

	float position[6] = {};
	int sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	struct sockaddr_in gcAddr;
	struct sockaddr_in locAddr;
	//struct sockaddr_in fromAddr;
	uint8_t buf[BUFFER_LENGTH];
	ssize_t recsize;
	socklen_t fromlen = sizeof(gcAddr);
	int bytes_sent;
	mavlink_message_t msg;
	uint16_t len;
	int i = 0;
	uint8_t rssi=0;
	uint8_t rem_rssi=0;
	uint8_t txbuf=0;
	uint8_t noise=0;
	uint8_t rem_noise=0;
	uint16_t rx_errors=0;
	int lat=0;
	int lon=0;
	int heading=0;
	int k=0;
	int file=0;
	char includeHeading=0;

	char* output_file;
	//int success = 0;
	unsigned int temp = 0;
	FILE*		filePtr;

	printf("date,time,rssi,remrssi,noise,remnoise,lat,lon,heading\n");

	for(k=1;k<argc;k++)
	{
		if(argv[k][0] == '-')
		{
			switch (argv[k][1])
			{
				//case 'h':usage();
					break;
				case 'f':output_file=argv[k+1];
					file=1;
					k++;
					break;
				case 'h':includeHeading=1;
					break;

				default:
					printf("Invalid Argument: %s\n", argv[k]);
					//usage();
			}
		}
		else{
			printf("\nInvalid Argument: %c%c\n", argv[k][0],argv[k][1]);
			exit(0);
		}
	}

	// Check if --help flag was used
	if ((argc == 2) && (strcmp(argv[1], help) == 0))
    {
		printf("\n");
		printf("\tUsage:\n\n");
		printf("\t");
		printf("%s", argv[0]);
		printf(" <ip address of QGroundControl>\n");
		printf("\tDefault for localhost: udp-server 127.0.0.1\n\n");
		exit(EXIT_FAILURE);
    }


	// Change the target ip if parameter was given
	strcpy(target_ip, "127.0.0.1");
	if (argc == 2)
    {
		strcpy(target_ip, argv[1]);
    }

	if(file){
		printf("Opening file: %s\n",output_file);
		filePtr = fopen(output_file,"w");

		if(filePtr==NULL){
			printf("Can't open file: %s! Exiting...\n", output_file);
			exit(0);
		}

		fprintf(filePtr,"date,time,rssi,remrssi,noise,remnoise,lat,lon,heading\n");

	}

	memset(&locAddr, 0, sizeof(locAddr));
	locAddr.sin_family = AF_INET;
	locAddr.sin_addr.s_addr = INADDR_ANY;
	//locAddr.sin_port = htons(14550);
	locAddr.sin_port = htons(14445);
	/* Bind the socket to port 14551 - necessary to receive packets from qgroundcontrol */
	if (-1 == bind(sock,(struct sockaddr *)&locAddr, sizeof(struct sockaddr)))
    {
		perror("error bind failed");
		close(sock);
		exit(EXIT_FAILURE);
    }

	/* Attempt to make it non blocking */
#if (defined __QNX__) | (defined __QNXNTO__)
	if (fcntl(sock, F_SETFL, O_NONBLOCK | FASYNC) < 0)
#else
	if (fcntl(sock, F_SETFL, O_NONBLOCK | O_ASYNC) < 0)
#endif

    {
		fprintf(stderr, "error setting nonblocking: %s\n", strerror(errno));
		close(sock);
		exit(EXIT_FAILURE);
    }


	memset(&gcAddr, 0, sizeof(gcAddr));
	gcAddr.sin_family = AF_INET;
	gcAddr.sin_addr.s_addr = inet_addr(target_ip);
	gcAddr.sin_port = htons(14445);

	for (;;)
    	{
		memset(buf, 0, BUFFER_LENGTH);
		recsize = recvfrom(sock, (void *)buf, BUFFER_LENGTH, 0, (struct sockaddr *)&gcAddr, &fromlen);
		if (recsize > 0)
      		{
			// Something received - print out all bytes and parse packet
			mavlink_message_t msg;
			mavlink_status_t status;

			//printf("Bytes Received: %d\nDatagram: ", (int)recsize);
			for (i = 0; i < recsize; ++i)
			{
				temp = buf[i];
				//printf("%02x ", (unsigned char)temp);
				if (mavlink_parse_char(MAVLINK_COMM_0, buf[i], &msg, &status))
				{
					// Packet received
					if(msg.msgid == 109){
						rssi=mavlink_msg_radio_status_get_rssi(&msg);
						rem_rssi=mavlink_msg_radio_status_get_remrssi(&msg);
						noise=mavlink_msg_radio_status_get_noise(&msg);
						rem_noise=mavlink_msg_radio_status_get_remnoise(&msg);

    						now = time (0);
    						strftime (buff, 100, "%Y%m%d,%H%M%S,", localtime (&now));

    						//printf ("%s", buff);
						//printf("%d,%d,%d,%d,%f,%f\n",rssi, rem_rssi,noise, rem_noise,lat/10000000.,lon/10000000.);
					}

					if (msg.msgid == MAVLINK_MSG_ID_GLOBAL_POSITION_INT )
                                 	{
                                         	mavlink_global_position_int_t global = {};
                                         	mavlink_msg_global_position_int_decode(&msg, &global);
						lat=global.lat;
						lon=global.lon;
						heading=global.hdg;
						printf ("%s", buff);

						if(includeHeading==1)
							printf("%d,%d,%d,%d,%f,%f,%6.3f\n",rssi, rem_rssi,noise, rem_noise,lat/10000000.,lon/10000000.,heading/1000.);
						else
							printf("%d,%d,%d,%d,%f,%f\n",rssi, rem_rssi,noise, rem_noise,lat/10000000.,lon/10000000.);
					    if(file && lat != 0 && lon != 0){
						fprintf (filePtr,"%s", buff);
						if(includeHeading)
							fprintf(filePtr,"%d,%d,%d,%d,%f,%f,%6.3f\n",rssi, rem_rssi,noise, rem_noise,lat/10000000.,lon/10000000.,heading/1000.);
						else
							fprintf(filePtr,"%d,%d,%d,%d,%f,%f\n",rssi, rem_rssi,noise, rem_noise,lat/10000000.,lon/10000000.);
                                  	    }
					}
				}
			}
			//printf("\n");
		}
		memset(buf, 0, BUFFER_LENGTH);
		usleep(50); // Sleep one second
    }
}


/* QNX timer version */
#if (defined __QNX__) | (defined __QNXNTO__)
uint64_t microsSinceEpoch()
{

	struct timespec time;

	uint64_t micros = 0;

	clock_gettime(CLOCK_REALTIME, &time);
	micros = (uint64_t)time.tv_sec * 1000000 + time.tv_nsec/1000;

	return micros;
}
#else
uint64_t microsSinceEpoch()
{

	struct timeval tv;

	uint64_t micros = 0;

	gettimeofday(&tv, NULL);
	micros =  ((uint64_t)tv.tv_sec) * 1000000 + tv.tv_usec;

	return micros;
}
#endif
