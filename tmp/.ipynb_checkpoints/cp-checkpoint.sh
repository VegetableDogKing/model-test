#while read line; do
    #name=$(echo $line | cut -c 43-57)
    #if [ ! -f $name ] ; then
        #echo $name
        #ffmpeg -i $line -q:v 0 $name 2>&/dev/null
    #fi
    #ffmpeg -i $line -q:v 0 $name
#done < t1.txt

name=14447279_03
rm ${name}.mp4
ffmpeg -i /Data/user_jchou/epilepsy/data/video_trim/$name.mp4 -q:v 0 ${name}.mp4