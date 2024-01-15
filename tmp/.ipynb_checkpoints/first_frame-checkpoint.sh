for mp4 in *.mp4; do
 ffmpeg -i $mp4 -vframes 1 -f image2 ./tmp/$mp4.jpg
done
