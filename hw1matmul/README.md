This repo is for HW1. This folder might have several test files other than the main HW file.

**hostfile**

This file has the IP address of the machines connected in the cluster.
The cluster is a set of 2 Fedora Linux (Fedora 37) machines connected via a switch.

**commands**
mpirun --hostfile hostfile -np 4 --map-by node matrix_mult 4000 3000 2000

>> this runs the matrix multiplication on two machines.



