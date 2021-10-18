#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate strangefish

export STOCKFISH_EXECUTABLE="/home/rbc/reconchess/strangefish-master/stockfish_14_x64"
export PYTHONPATH="/home/rbc/reconchess/strangefish-master/.":$PYTHONPATH
export PYTHONPATH="/home/rbc/reconchess/Fianchetto/.":$PYTHONPATH
export PYTHONPATH="/home/rbc/reconchess/fishstrange/.":$PYTHONPATH

rm *.json
rm -rf attacker_games
mkdir attacker_games
cd attacker_games
cp ../weights_run3_752050.pb.gz ./
for i in {1..4} # 2 hour for 4 games
do
  for j in {1..15} # no games per 1/2 hour per gpu
  do
	  python3 ../scripts/attacker_w.py 0 &> white_$i\_$j.log & # gpu 0
	  pids[2*j-2]=$!
	  python3 ../scripts/attacker.py 1 &> black_$i\_$j.log & # gpu 1
	  pids[2*j-1]=$!
  done
  # wait for all pids
  for pid in ${pids[*]}; do
	  wait $pid
	  echo $pid "done"
  done
done
#
#
# for i in {1..15}
# do
# 	python3 scripts/attacker_w.py &
# 	python3 scripts/attacker.py
# done
