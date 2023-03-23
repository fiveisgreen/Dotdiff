import numpy as np
import cv2

### Controls ###
save_output_video = False
frames_until_break = -1 #quit after playing this many frames. Set to -1 to play until end, or quit command 'q'
frames_until_pause = 0 #pause after playing this many frames. Set to -1 to not pause (unless a pause command is given.)
window_name = "frame"

#Params for text overlay:
text_position = (100,100)
text_overlay = "frame {} {:.1f}%"

#Params for the rectangle overlay
rec_start_point = (1065,262)
rec_end_point = (1404,481)
rec_color = (0,255,0) #BGR color. a 4th and final param is possible here. If it's alpha, it doesn't work.
rec_thickness= 2
### End Controls ###

### Functions ###
def keyboard_logic(): #logic of the video quit and pause mode.
    global frames_until_pause 
    quit_keys = (ord('q'), 27) #These characters trigger video quit.  27 is the ASCII value of the escape key
    pause_keys = (ord('p'), ord(' ')) #These characters trigger pause, unpause.
    if frames_until_pause > 0:
        frames_until_pause -= 1
    keylogged = cv2.waitKey(1) & 0xFF
    if keylogged in quit_keys: # mid-video quitting
        return True
    if frames_until_pause == 0 or keylogged in pause_keys: #pause
        if frames_until_pause == 0:
            frames_until_pause = -1
        while(True): #loop while paused, doing ~nothing but keyWait
            keylogged = cv2.waitKey(50) & 0xFF
            if keylogged in quit_keys: #mid-pause quit
                return True
            elif keylogged in pause_keys: #unpause
                return False
            elif keylogged >= ord('0') and keylogged <= ord('9'): #While paused, num key n-> advance 2^n frames.
                N = keylogged - ord('0')
                frames_until_pause = 2**N 
                return False

### Keyboard Instructions ###
print("""USER GUIDE:
Press 'q' or ESC to exit video, 'p' or space to pause. While paused, pressing any number n will advance 2^n frames, then pause again. In the controls, you can the displayed video by setting save_output_video = True.  The video can be made to automatically quit after n frames by setting frames_until_break = n. Set that to -1 to play until end, or quit command 'q'.  The video can be made to automatically pause after n frames by setting frames_until_pause = n. n=0 is often useful to start paused. Set that to -1 to not pause. 
Note: the lower left corner shows useful pixel coordinates and values.\n""")

### Video IO and Window Setup ###
cap = cv2.VideoCapture('2023-03-16 11-11-07.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')#MP4V gives warnings.... seriously
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
#For more Cap_PROP tags see https://www.geeksforgeeks.org/how-to-get-properties-of-python-cv2-videocapture-object/
print(f"frame width: {frame_width}, height: {frame_height}, fps: {fps:.2f}, n frames: {total_frames }")
frame_dims = (frame_width, frame_height)
if save_output_video:
    out = cv2.VideoWriter('output.mp4',fourcc, fps, frame_dims)

cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN) 
#Options list: https://docs.opencv.org/4.x/d0/d90/group__highgui__window__flags.html#ggabf7d2c5625bc59ac130287f925557ac3acf621ace7a54954cbac01df27e47228f
#initial black frame for moveWindow to work with. 
cv2.imshow(window_name,np.zeros(shape=(frame_height, frame_width, 3)).astype('uint8')) 
cv2.moveWindow(window_name, 1,1)  

nframe = -1 
while(cap.isOpened()):
    nframe += 1
    ret, in_frame = cap.read() #bool ret; numpy.ndarray frame. 
    #print(type(in_frame), in_frame.shape, in_frame.dtype) #type(in_frame) = <class 'numpy.ndarray'> , in_frame.shape = (1080, 1920, 3), in_frame.dtype = uint8
    if not ret or (frames_until_break >= 0 and nframe >= frames_until_break):
        break

    #### CV and DECORATIONS ####
    out_frame = cv2.rectangle(in_frame, rec_start_point, rec_end_point, rec_color, rec_thickness)
    cv2.putText(out_frame, 
     text_overlay.format(nframe, 100.0*nframe/total_frames),
     text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, #font size
     rec_color, #font color
     3) #font stroke
    #putText guide: https://pythonexamples.org/python-opencv-write-text-on-image-puttext/

    cv2.imshow(window_name, out_frame)
    if save_output_video:
        out.write(out_frame)
    if keyboard_logic():
        break

#Finish up
cap.release()
if save_output_video:
    out.release()
cv2.destroyAllWindows()

##### NOTES and useful links ######
#drawing lines: https://www.geeksforgeeks.org/python-opencv-cv2-line-method/
#drawing text: https://pythonexamples.org/python-opencv-write-text-on-image-puttext/
#to gray scale: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
#val_thresholding: https://stackabuse.com/opencv-val_thresholding-in-python-with-cv2val_threshold/
#val_thresholding options: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_val_thresholding.html

#real magic: the blob detector: https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

#https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html

#cropping a section of an image: something like 
    #cr = crop(1,1,64,67)
    #cropped_img = image(cr)
