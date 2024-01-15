# https://pixspy.com/
id="28880877_04"
x=40
y=20
x2=430
y2=480
out_w=$(( $x2-$x ))
out_h=$(( $y2-$y ))
ffmpeg -i $id.mp4 -filter:v "crop=$out_w:$out_h:$x:$y" ../tmp-crop/$id.mp4