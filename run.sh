cd demo/src
rm -rf frames

cd ../transfer_data
rm ./*

cd ../results
rm -rf ./*

cd ../..

cd generate_video

python get_keyframes.py
python get_human_body_info.py
python generate_video.py

cd ../demo

ffmpeg -i ./results/frames/pred_%06d.jpg -vcodec mpeg4 ./results/demo.mp4
