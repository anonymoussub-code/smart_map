#!/bin/sh
./rijndael input_large.asc output_large.enc e 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321
./rijndael output_large.enc output_large.dec d 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321

