from math import sqrt

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
            Y2sum     = float(sum([y*y for y in Y]) ) #0.078
            self.Xmin = float(min(X) )
            self.Xmax = float(max(X) )
            Xsum      = float(sum(X) )
            X2sum     = float(sum([x*x for x in X]) ) #0.077
            self.Wmin = float(min(W) )
            self.Wmax = float(max(W) )
            Wsum      = float(sum(W) )
            W2sum     = sum([float(v)**2 for v in W]) #0.4 s 
            WXsum     = float(sum([x*w for x,w in zip(X,W)]) ) #slow 0.38s :: 60frames
            WYsum     = float(sum([y*w for y,w in zip(Y,W)]) ) #slow 0.38s
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
                stdev_factor_radical = 1.0 - (float(W2sum - 2*self.Wmin*Wsum + N*(self.Wmin*self.Wmin) )/stdev_factor_deonominator )
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

def look_at_neighbors_swiss(img, center, minval, imgW, imgH):
    #add pixels to todo_list, in a swiss cross, pattern, if their values are >= minval
    #constrained to be in the image.
    #returns a todo list segment whose elements are tuples ( ( y, x), pixel_value )
    y,x = center
    validity = (x < imgW-1, x > 0, y < imgH-1, y > 0) 
    coords = ( (y,x+1), (y,x-1), (y+1,x), (y-1,x))
    return [ coords[i] for i in range(4) if validity[i] and img[coords[i]] >= minval]
            
    """ 
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
        #c.register(center, img[center]) #notice the todo list doesn't need pixel values, only reg does. 
        Y.append(center[0])
        X.append(center[1])
        W.append(img[center])
        N+=1
        if N >= 4096: #if we're over-clustering, break
            c = Cluster(X, Y, W, N)
            c.N = -c.N
            return c
        new_items = look_at_neighbors_swiss(img, center, minval, imgW, imgH)
        todo_list.extend(new_items )
        for item in new_items:
            img[item] = cluster_color 
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
