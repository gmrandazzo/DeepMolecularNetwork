#!/usr/bin/env bash
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details
export AMBERHOME=/opt/amber/amber18/
source $AMBERHOME/amber.sh

atomtype -f mol2 -i $1 -o $1.ac -a 1
antechamber -fi ac -i $1.ac -fo mol2 -o $2 -at sybil
