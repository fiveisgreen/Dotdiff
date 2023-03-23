import numpy as np
import cv2
import cluster as bc
import sys
import os.path


#TODO: 
#* set frame position with cap.set(cv2.cAP_PROP_POS_FRAMES, frame_position)
#* add mouse click calibration


### Controls ###
save_output_video = False
frames_until_break = -1 #quit after playing this many frames. Set to -1 to play until end, or quit command 'q'
frames_until_pause = 0 #pause after playing this many frames. Set to -1 to not pause (unless a pause command is given.)
n_frames_to_skip = 15
window_name = "frame"
value_threshold = 200 #93
saturation_threshold = 65
"""
upper_cluster_frame0_coords = (362, 1227) #Y, X 
lower_cluster_frame0_coords = (953, 1276) #Y, X 
input_video_file_name = '2023-03-16 11-11-07.mp4'
"""

upper_cluster_frame0_coords = (219, 893) #Y, X 
lower_cluster_frame0_coords = (945, 934) #Y, X 
input_video_file_name = '2023-03-20 21-03-27 BdotsOnW.mp4'
if len(sys.argv) > 1:
    os.path.isfile(sys.argv[1])
    input_video_file_name = sys.argv[1]
if not os.path.isfile(input_video_file_name):
    print(f"File {input_video_file_name} not found.")
    sys.exit()

#input_video_file_name = '2023-03-21 11-19-42 dont process.mp4'
#input_video_file_name = 'dots on black mask.mp4'

output_video_file_name = '2023-03-20 21-03-27 BdotsOnW CVify.mp4'
cluster_color = 50 #ought to be significantly less than value_threshold

#Params for text overlay:
text_position = (100,100) #x,y
text_position2 = (100,200) #x,y
text_position3 = (100,300) #x,y
text_position4 = (100,400) #x,y
text_overlay = "Frame {} {:.1f}%"
dmdm_thresh = 1.08
text_overlay3 = "Threshold: {:.2f}".format(dmdm_thresh)

#Params for the overlay
overlay_color = (0,255,0) #BGR color. a 4th and final param is possible here. If it's alpha, it doesn't work.
nope_color = (255,255,255)
overlay_line_width_pixels= 2
### End Controls ###

### Functions ###
def YX2XY(coords):
    return (coords[1], coords[0])

def XY2YX(coords):
    return YX2XY(coords)

def mouse_click_logic(event, x, y, flags, param):
    #call with cv2.setMouseCallback(window_name, mouse_click_logic)
    #for a list of available callback events, see: 
    #https://docs.opencv.org/3.4/d0/d90/group__highgui__window__flags.html#ga927593befdddc7e7013602bca9b079b0
    #cv.EVENT_RBUTTONUP indicates that right mouse button is released. Name convention holds R/L, UP, DOWN
    #cv.EVENT_LBUTTONDBLCLK indicates that left mouse button is double clicked.
    #cv.EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling, respectively.
    if event == cv2.EVENT_LBUTTONDOWN:  
        print(f"Mouse clicked x,y = {x}, {y}")
        return (y,x)

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
            cv2.setMouseCallback(window_name, mouse_click_logic)
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
cap = cv2.VideoCapture(input_video_file_name)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')#MP4V gives warnings.... seriously
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
#For more Cap_PROP tags see https://www.geeksforgeeks.org/how-to-get-properties-of-python-cv2-videocapture-object/
print(f"frame width: {frame_width}, height: {frame_height}, fps: {fps:.2f}, n frames: {total_frames }")
frame_dims = (frame_width, frame_height)
if save_output_video:
    out = cv2.VideoWriter(output_video_file_name ,fourcc, fps, frame_dims)

cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN) 
#Options list: https://docs.opencv.org/4.x/d0/d90/group__highgui__window__flags.html#ggabf7d2c5625bc59ac130287f925557ac3acf621ace7a54954cbac01df27e47228f
#initial black frame for moveWindow to work with. 
cv2.imshow(window_name,np.zeros(shape=(frame_height, frame_width, 3)).astype('uint8')) 
cv2.moveWindow(window_name, 1,1)  

upper_cluster_last_coords = upper_cluster_frame0_coords #y,x
lower_cluster_last_coords = lower_cluster_frame0_coords 
dmin = frame_width
dmax = 0
intercluster_distance_pixels = frame_width
nframe = -1 
while(cap.isOpened()):
    nframe += 1
    ret, in_frame = cap.read() #bool ret; numpy.ndarray in_frame. 
    if nframe < n_frames_to_skip:
        continue
    #print(type(in_frame), in_frame.shape, in_frame.dtype) #type(in_frame) = <class 'numpy.ndarray'> , in_frame.shape = (1080, 1920, 3), in_frame.dtype = uint8
    if not ret or (frames_until_break >= 0 and nframe >= frames_until_break):
        break

    ### CV ###
    #Thresholding of inverted image. Black dots -> white. junk->black
    inv_thresh = 255-value_threshold
    _, saturation_map, value_map = cv2.split(cv2.cvtColor(in_frame, cv2.COLOR_BGR2HSV)) 
    _, saturation_map = cv2.threshold(saturation_map, saturation_threshold, 255, cv2.THRESH_BINARY_INV) #now saturation_map is really a mask
    _, value_map = cv2.threshold(~value_map, inv_thresh, 255, cv2.THRESH_TOZERO)
    #first return value of cv2.threshold is the threshold 
    #manual: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    value_map = cv2.bitwise_and(value_map, saturation_map)
    #print(type(in_frame), in_frame.shape, in_frame.dtype)
    #print(type(saturation_map), saturation_map.shape, saturation_map.dtype)
    #<class 'numpy.ndarray'> (1080, 1920, 3) uint8 -- from threshold orig
    #<class 'numpy.ndarray'> (1080, 1920) uint8 -- from thresh cvtColor
    #<class 'numpy.ndarray'> (1080, 1920) uint8 -- from split
    ### CLUSTERING ###

    max_seek_radius = int(max(100,0.65*intercluster_distance_pixels))
    foundU, upper_cluster_seed = bc.seek(value_map, upper_cluster_last_coords, inv_thresh, frame_width, frame_height, max_seek_radius )
    foundL, lower_cluster_seed = bc.seek(value_map, lower_cluster_last_coords, inv_thresh, frame_width, frame_height, max_seek_radius )
    
    if foundU:
        cluster_upper = bc.cluster(value_map, upper_cluster_seed, inv_thresh, frame_width, frame_height, cluster_color )
        upper_cluster_last_coords = cluster_upper.get_center() 

    if foundL:
        cluster_lower = bc.cluster(value_map, lower_cluster_seed, inv_thresh, frame_width, frame_height, cluster_color )
        lower_cluster_last_coords = cluster_lower.get_center() 

    #DEBUG
    out_frame = in_frame
    #out_frame = cv2.cvtColor(value_map, cv2.COLOR_GRAY2BGR)
    #out_frame = cv2.cvtColor(saturation_map, cv2.COLOR_GRAY2BGR)

    #### DECORATIONS ####
    if foundU and foundL and cluster_upper.N > 2 and cluster_lower.N > 2:
        intercluster_distance_pixels = cluster_upper.distance_to(cluster_lower)
        #TODO make printing this optional 
        #    print(f"   cluster_upper N={cluster_upper.N }, cluster_lower N={cluster_lower.N}")
        #    print(f"   cu = {upper_cluster_last_coords[0] },{upper_cluster_last_coords[1] }, cl = {lower_cluster_last_coords[0] },{lower_cluster_last_coords[1] }, dist = {intercluster_distance_pixels:.1f} ")

        dmin = min(dmin,intercluster_distance_pixels)
        dmax = max(dmax,intercluster_distance_pixels)
        dmdm = dmax/dmin
        cv2.line(out_frame, YX2XY(upper_cluster_last_coords), YX2XY(lower_cluster_last_coords), overlay_color, overlay_line_width_pixels)
        cv2.putText(out_frame, f"Dist: {intercluster_distance_pixels:.1f} pixels",
         text_position2, cv2.FONT_HERSHEY_SIMPLEX, 2, overlay_color, overlay_line_width_pixels) 
        if dmdm > dmdm_thresh:
            dm_color = overlay_color
        else:
            dm_color = nope_color
        cv2.putText(out_frame, f"Current Dmax/Dmin: {dmdm:.3f}",
         text_position4, cv2.FONT_HERSHEY_SIMPLEX, 2, dm_color, overlay_line_width_pixels) 

        #Threshold: 1.08
        #Current Max: 1.01 (turns white if < thresh, turns green if > thresh)

    cv2.putText(out_frame, 
     text_overlay.format(nframe, 100.0*nframe/total_frames),
     text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, #font size in some weird units
     overlay_color, overlay_line_width_pixels) #font stroke
    #putText guide: https://pythonexamples.org/python-opencv-write-text-on-image-puttext/

    cv2.putText(out_frame, text_overlay3, text_position3, cv2.FONT_HERSHEY_SIMPLEX, 2, overlay_color, overlay_line_width_pixels) #font stroke


    cv2.imshow(window_name, out_frame)
    if save_output_video:
        out.write(out_frame)
    if keyboard_logic():
        print(f"Quit at frame {nframe }")
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
#value_thresholdolding: https://stackabuse.com/opencv-value_thresholdolding-in-python-with-cv2value_thresholdold/
#value_thresholdolding options: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_value_thresholdolding.html
#rectangles:
    #out_frame = cv2.rectangle(out_frame, rec_start_point, rec_end_point, overlay_color, overlay_line_width_pixels)
#circles: cv2.circle(img, center, radius, something, line_width?)

#real magic: the blob detector: https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

#https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html

#cropping a section of an image: something like 
    #cr = crop(1,1,64,67)
    #cropped_img = image(cr)
#keep in mind the selectROIs function for drawing constraint boxes on the image. 

