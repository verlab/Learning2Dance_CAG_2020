#!/bin/bash

export CUDA_CACHE_PATH= $TMP_OPENPOSE /tmp/openpose_thiago/
cd $OPENPOSE_PATH

N=1 ## Number of parallel OpenPose instances to run.

for instance in `ls $1`;
do
	(
    for file in `ls $1$instance/`;
	do
		if [[ "$file" != *.wav ]]
		then
			if [[ "$file" != *.json ]]
			then
				echo "Calculating for $1$instance/$file/$file.mp4"
				mkdir -p $1$instance/$file/openpose/json/
				mkdir -p $1$instance/$file/openpose/rendered_poses/
				#CALL OPENPOSE HERE
				./build/examples/openpose/openpose.bin --video $1$instance/$file/$file.mp4 --write_json $1$instance/$file/openpose/json/ --write_video $1$instance/$file/openpose/rendered_poses/$instance.avi --display 0 --render_pose 1
			fi
		fi
	done
    ) &
	if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
		wait -n
	fi
done
