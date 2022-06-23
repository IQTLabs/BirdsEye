#!/bin/bash

set -e
infile=$1

if [ ! -f "${infile}" ] ; then
  echo extract subtiles from mp4 to .srt file.
  echo usage: $0 mp4file
  exit 1
fi

outfile=$(echo $infile|sed 's/.mp4/.srt/i')
ffmpeg -loglevel -8 -i "${infile}" -map 0:s:0 -y "${outfile}"
echo extracted to ${outfile}
