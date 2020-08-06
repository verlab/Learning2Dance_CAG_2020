DATASET_DIR=$1

N=3

for style in `ls $DATASET_DIR`;
do
    (
    for video in `ls $DATASET_DIR$style/`;
    do
    #ls | grep "I_[0-9]$"
        if test -f "$DATASET_DIR$style/$video/$video.mp4"; then
            ((i=i%N)); ((i++==0)) && wait
            echo "Rendering video for $DATASET_DIR$style/$video/$video.mp4"
            python render_data.py --test_dataset 0 --render_video $DATASET_DIR$style/$video &
        fi
    done
    )
done
