ffmpeg -y -start_number 1 -i 'data/render_final/%d.png'  -i data/bad_apple.wav -c:v libx265 -b:v 10000k -r 29.97 output.mp4