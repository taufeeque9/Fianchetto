#!/bin/bash

export STOCKFISH_EXECUTABLE="/home/taufeeque/reconchess/strangefish-master/stockfish_14_x64"
export PYTHONPATH="/home/taufeeque/reconchess/strangefish-master/.":$PYTHONPATH
export PYTHONPATH="/home/taufeeque/reconchess/Fianchetto/.":$PYTHONPATH
export PYTHONPATH="/home/taufeeque/reconchess/fishstrange/.":$PYTHONPATH

rm *.json

for i in {1..15}
do
	python3 scripts/white.py
	python3 scripts/black.py
done
