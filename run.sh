cd demo/results
rm -rf ./*

cd ../transfer_data
rm ./*

cd ..

cd ../get_keyframe/frames
rm ./*

cd ..
python demo.py

cd ../get_human_body_info/demo_output
rm -rf ./*
cd ..
python demo_video.py

cd ../generate_video
python demo.py --gpu_ids 0