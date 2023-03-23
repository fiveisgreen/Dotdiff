# Dotdiff
A computer vision library for tracking dots

## Environment Setup
In any python environment:
pip install opencv-python

## Running:
There are two scripts of importance: a basically blank computer vision video playground called helloworld.py
$ python3 helloworld.py

And the main program: dotdiff.py. This can use either hard-coded input video file names, or they can be supplied to the first command line argument. 
$ python dotdiff.py
$ python dotdiff.py "myInputVideo.mp4"

Currently only mp4 videos are supported.

## User Guide
Press 'q' or ESC to exit video, 'p' or space to pause. While paused, pressing any number n will advance 2^n frames, then pause again. In the controls, you can the displayed video by setting save\_output\_video = True.  The video can be made to automatically quit after n frames by setting frames\_until\_break = n. Set that to -1 to play until end, or quit command 'q'.  The video can be made to automatically pause after n frames by setting frames\_until\_pause = n. n=0 is often useful to start paused. Set that to -1 to not pause.
Note: the lower left corner shows useful pixel coordinates and values.

(test)
