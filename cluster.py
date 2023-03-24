from math import sqrt
from functools import reduce
import numpy as np

"""
optomization: 
    Cluster and look_at_neighbors_swiss are the slow parts. Cluster.__init__ is pretty bad too, particularly in the list comprehensions. 
"""

class Cluster:
    def __init__(self, X,Y,W,N):
        self.N = N
        if N <=0:
            self.Ymin = -1 
            self.Ymax = -1
            self.Xmin = -1 
            self.Xmax = -1
            self.Wmin = -1
            self.Wmax = -1
            self.Yavg = -1 
            self.Xavg = -1
            self.Wavg = -1
            self.Ystdev = -1
            self.Xstdev = -1
            self.Wstdev = -1
            self.YWavg = -1
            self.XWavg = -1
            self.YWstdev = -1
            self.XWstdev = -1
        else:
            #note trouble with ubyte overflow from W
            self.Ymin = float(min(Y) )
            self.Ymax = float(max(Y) )
            Ysum      = float(sum(Y) )
            Y2sum     = float(np.sum(np.square(Y)) ) #fastest over list comprehension, generators, and np.dot
            self.Xmin = float(min(X) )
            self.Xmax = float(max(X) )
            Xsum      = float(sum(X) )
            X2sum     = float(np.sum(np.square(X)))
            self.Wmin = float(min(W) )
            self.Wmax = float(max(W) )
            Wnp = np.array(W, dtype=np.uint32)
            Wsum  = float(np.sum(Wnp))
            W2sum = float(np.sum(np.square(Wnp)))
            WXsum     = float(np.dot(X,W) )
            WYsum     = float(np.dot(Y,W) )
            ####
            Nfloat = float(N)
            self.Yavg = Ysum/Nfloat
            self.Xavg = Xsum/Nfloat
            self.Wavg = Wsum/Nfloat
            self.Ystdev = sqrt( (Y2sum/Nfloat) - self.Yavg*self.Yavg )
            self.Xstdev = sqrt( (X2sum/Nfloat) - self.Xavg*self.Xavg )
            self.Wstdev = sqrt( (W2sum/Nfloat) - self.Wavg*self.Wavg )

            Wdiff = float(Wsum - N*self.Wmin)
            stdev_factor_deonominator = Wdiff*Wdiff
            if stdev_factor_deonominator > 0:
                self.YWavg = float(WYsum - self.Wmin*Ysum)/Wdiff
                self.XWavg = float(WXsum - self.Wmin*Xsum)/Wdiff
                stdev_factor_radical = 1.0 - (float(W2sum - self.Wmin*Wsum*2 + N*(self.Wmin*self.Wmin) )/stdev_factor_deonominator )
                #print(f"N {N}, stdev_factor_radical {stdev_factor_radical }, Wmin {self.Wmin}, Wmax {self.Wmax}, Wsum {Wsum} W2sum {W2sum}, deominator {stdev_factor_deonominator}")
                stdev_factor = 1 
                if stdev_factor_radical > 0:
                    stdev_factor = sqrt(stdev_factor_radical)
                self.XWstdev = self.Xstdev*stdev_factor
                self.YWstdev = self.Ystdev*stdev_factor
                #self.validW = True
            else: #This happens when all weights are identical.
                self.YWavg = self.Yavg 
                self.XWavg = self.Xavg 
                self.XWstdev = self.Xstdev
                self.YWstdev = self.Ystdev
                #self.validW = False
        
    def get_center(self,use_weighted_averages = False):
        #returns (y,x) coordinates. y,x because that can be used to index the image and get pixel values.
        if use_weighted_averages:
            return (round(self.YWavg), round(self.XWavg))
        else:
            return (round(self.Yavg), round(self.Xavg))
    def distance_to(self, cluster2, use_weighted_averages = True):
        if use_weighted_averages:
            return sqrt( (self.YWavg - cluster2.YWavg)**2 + (self.XWavg - cluster2.XWavg)**2) 
        else:
            return sqrt( (self.Yavg - cluster2.Yavg)**2 + (self.Xavg - cluster2.Xavg)**2) 

def look_at_neighbors_swiss(img, todo_list, center, minval, imgW, imgH, color):
    #add pixels to todo_list, in a swiss cross, pattern, if their values are >= minval
    #constrained to be in the image.
    #returns a todo list segment whose elements are tuples ( ( y, x), pixel_value )
    y,x = center
    xp = x+1
    xm = x-1
    yp = y+1
    ym = y-1
    if xp < imgW and img[y, xp] >= minval:
        todo_list.append((y,xp))
        img[y,xp] = color 
    if x > 0 and img[y,xm] >= minval:
        todo_list.append((y,xm))
        img[y,xm] = color 
    if yp < imgH and img[yp,x] >= minval:
        todo_list.append((yp,x))
        img[yp,x] = color 
    if y > 0 and img[ym,x] >= minval:
        todo_list.append((ym,x))
        img[ym,x] = color 

    """
    y,x = center
    if x < imgW-1 and img[y, x+1] >= minval:
        todo_list.append((y,x+1))
        img[y,x+1] = cluster_color 
    if x > 0 and img[y,x-1] >= minval:
        todo_list.append((y,x-1))
        img[y,x-1] = cluster_color 
    if y < imgH-1 and img[y+1,x] >= minval:
        todo_list.append((y+1,x))
        img[y+1,x] = cluster_color 
    if y > 0 and img[y-1,x] >= minval:
        todo_list.append((y-1,x))
        img[y-1,x] = cluster_color 


    y,x = center
    coords = ( (y,x+1), (y,x-1), (y+1,x), (y-1,x) )
    if x < imgW-1 and img[coords[0]] >= minval:
        todo_list.append(coords[0])
        img[coords[0]] = cluster_color 
    if x > 0 and img[coords[1]] >= minval:
        todo_list.append(coords[1])
        img[coords[1]] = cluster_color 
    if y < imgH-1 and img[coords[2]] >= minval:
        todo_list.append(coords[2])
        img[coords[2]] = cluster_color 
    if y > 0 and img[coords[3]] >= minval:
        todo_list.append(coords[3])
        img[coords[3]] = cluster_color 

    
    y,x = center
    validity = (x < imgW-1, x > 0, y < imgH-1, y > 0) 
    coords = ( (y,x+1), (y,x-1), (y+1,x), (y-1,x) )
    #return [ coords[i] for i in range(4) if validity[i] and img[coords[i]] >= minval]
    for i in range(4):
        if validity[i] and img[coords[i]] >= minval:
            todo_list.append(coords[i])
            img[coords[i]] = cluster_color 

    #47ms/F correct.
    new_todo = [] #a fifo
    validity = (center[1] < imgW-1, center[1] > 0, center[0] < imgH-1, center[0] > 0) #E, W, N, S
    coords = ( (center[0],center[1]+1),
            (center[0],center[1]-1),
            (center[0]+1,center[1]),
            (center[0]-1,center[1]))
    for i in range(4):
        if validity[i]:
            ival = img[coords[i]]
            if is_cluster(ival, minval):
                new_todo.append((coords[i], ival))
    return new_todo
    """
        
def cluster(img, seed, minval, imgW, imgH, cluster_color ):
        #expect img[y][x] so max indicies are img[imgH-1][imgW-1]
        #accepted pixels must be light colored, with value >= minval
    cluster_color = max(0,min(cluster_color, minval-1))  #new color of clustered pixels, must be less than minval (~162)
    Y = []
    X = []
    W = []
    N = 0
    v = img[seed]
    if v < minval: #if not is_cluster(v, minval): #if seed isn't a candidate 
        return Cluster(X, Y, W, N)
    todo_list = [ seed ] # a fifo
    img[seed] = cluster_color  
    while todo_list:
        center = todo_list[0] #get front item
        Y.append(center[0])
        X.append(center[1])
        W.append(img[center])
        N+=1
        if N >= 4096: #if we're over-clustering, break
            c = Cluster(X, Y, W, N)
            c.N = -c.N
            return c
        look_at_neighbors_swiss(img, todo_list, center, minval, imgW, imgH, cluster_color)
        del todo_list[0]
    return Cluster(X, Y, W, N)

"""
def cluster(img, seed, minval, imgW, imgH, cluster_color ):
        #expect img[y][x] so max indicies are img[imgH-1][imgW-1]
        #accepted pixels must be light colored, with value >= minval
    cluster_color = max(0,min(cluster_color, minval-1))  #new color of clustered pixels, must be less than minval (~162)
    c = Cluster(imgW, imgH)
    v = img[seed]
    if v < minval: #if not is_cluster(v, minval): #if seed isn't a candidate 
        c.finalize()
        return c
    todo_list = [ (seed, v) ] # a fifo
    img[seed] = cluster_color  
    while len(todo_list) > 0:
        center = todo_list[0]
        c.register(center) #notice the todo list doesn't need pixel values, only reg does. 
        new_items = look_at_neighbors_swiss(img, center[0], minval, imgW, imgH)
        todo_list.extend(new_items )
        for item in new_items:
            img[item[0]] = cluster_color 
        del todo_list[0]
        if c.N >= 4096: #if we're over-clustering, break
            c.finalize()
            c.N = -c.N
            return c
    c.finalize()
    return c
"""

def seek(img, seed_guess, minval, imgW, imgH, max_radius):
   #returns bool success in finding seed, and the new seed. If no seed is found, return false, (-1,-1)
   #seed_guess is (y,x) coordinates so img can be index by it.
   if img[seed_guess] >= minval:
        return True, seed_guess
   step = 5
   for R in range(step, max_radius+1, step):
       #limit values of the squares for this Radius R
       y0 = max(0,seed_guess[0] - R)
       x0 = max(0,seed_guess[1] - R)
       y1 = min(imgH, seed_guess[0] + R + 1)
       x1 = min(imgW, seed_guess[1] + R + 1)

       #Look North
       y = y0
       for x in range(x0, x1, step):
            #if is_cluster(img[y][x], minval):
            if img[y][x] >= minval:
                return True, (y,x)
       #Look South    
       y = y1-1
       for x in range(x0, x1, step):
            #if is_cluster(img[y][x], minval):
            if img[y][x] >= minval:
                return True, (y,x)
       #Look East
       x = x0
       for y in range(y0, y1, step):
            #if is_cluster(img[y][x], minval):
            if img[y][x] >= minval:
                return True, (y,x)
       #Look West
       x = x1-1
       for y in range(y0, y1, step):
            #if is_cluster(img[y][x], minval):
            if img[y][x] >= minval:
                return True, (y,x)
   print("seek fails around seed ",seed_guess)
   return False, (-1,-1)


"""
print("begin test")
c = Cluster(1920, 1200)
c.register( ((9,8),7) )
c.register( ((9,7),7) )
c.register( ((7,5),4) )
c.register( ((1,2),1) )
c.finalize()
print(c.get_center(True), c.N, c.validW)
print(c.get_center(False), c.N, c.validW)
print("end test")
"""
