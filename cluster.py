from math import sqrt

def is_cluster(val, minval):
    return val >= minval

class Cluster:
    class __Sums:
        def __init__(self):
            self.Ysum = 0
            self.Y2sum = 0
            self.Xsum = 0
            self.X2sum = 0
            self.WXsum = 0
            self.WYsum = 0
            self.XYsum= 0
            self.WXYsum = 0
            self.Wsum = 0
            self.W2sum = 0
    def __init__(self, imgW, imgH):
        self.N = 0
        self.Ymin = imgH
        self.Ymax = 0
        self.Xmin = imgW
        self.Xmax = 0
        self.Wmin = 256
        self.Wmax = 0
        self.Yavg = -1 
        self.YWavg = -1
        self.Ystdev = -1
        self.YWstdev = -1
        self.Xavg = -1
        self.XWavg = -1
        self.Xstdev = -1
        self.XWstdev = -1
        self.Wavg = -1
        self.Wstdev = -1
        #self.validW = False
        self.S = self.__Sums()
    def register(self, pix): #pixels come as ( (y,x), pixel_value). Cannot be called after finalize()
        y,x = pix[0] 
        v = pix[1]
        self.N += 1
        self.Ymin = min(y, self.Ymin)
        self.Ymax = max(y, self.Ymax)
        self.Xmin = min(x, self.Xmin)
        self.Xmax = max(x, self.Xmax)
        self.Wmin = min(v, self.Wmin)
        self.Wmax = max(v, self.Wmax)
        self.S.Ysum  += y 
        self.S.Y2sum += y**2
        self.S.Xsum  += x
        self.S.X2sum += x**2
        self.S.Wsum  += v
        self.S.W2sum += v**2
        self.S.WYsum += y*v
        self.S.WXsum += x*v 
        #self.S.XYsum += y*x #maybe add xy covariance to the final cluster.
        #self.S.WXYsum+= y*x*v
    def finalize(self):
        if self.N > 0:
            Nfloat = float(self.N)
            self.Yavg = float(self.S.Ysum)/Nfloat
            self.Xavg = float(self.S.Xsum)/Nfloat
            self.Wavg = float(self.S.Wsum)/Nfloat
            self.Ystdev = sqrt( (float(self.S.Y2sum)/Nfloat) - self.Yavg**2 )
            self.Xstdev = sqrt( (float(self.S.X2sum)/Nfloat) - self.Xavg**2 )
            self.Wstdev = sqrt( (float(self.S.W2sum)/Nfloat) - self.Wavg**2 )

            Wdiff = float(self.S.Wsum - self.N*self.Wmin)
            stdev_factor_deonominator = Wdiff**2
            if stdev_factor_deonominator > 0:
                self.YWavg = float(self.S.WYsum - self.Wmin*self.S.Ysum)/Wdiff
                self.XWavg = float(self.S.WXsum - self.Wmin*self.S.Xsum)/Wdiff
                stdev_factor_radical = 1.0 - (float(self.S.W2sum - 2*self.Wmin*self.S.Wsum + self.N*(self.Wmin**2) )/stdev_factor_deonominator )
                #print(f"N {self.N}, stdev_factor_radical {stdev_factor_radical }, Wmin {self.Wmin}, Wmax {self.Wmax}, Wsum {self.S.Wsum} W2sum {self.S.W2sum}, deominator {stdev_factor_deonominator}")
                #stdev_factor_radical -3.5555555555555554, Wmin 4, Wmax 7, Wsum 11 W2sum 65, deominator 9.0
                    #wdiff = 11 - 2*4  =3
                    #denom = 9
                stdev_factor = 1
                if stdev_factor_radical > 0:
                    stdev_factor = sqrt(stdev_factor_radical)
                #stdev_factor = sqrt(1.0 - (float(self.S.W2sum - 2*self.Wmin*self.S.Wsum + (self.N*self.Wmin)**2)/stdev_factor_deonominator ))
                self.XWstdev = self.Xstdev*stdev_factor
                self.YWstdev = self.Ystdev*stdev_factor
                #self.validW = True
            else: #This happens when all weights are identical.
                self.YWavg = self.Yavg 
                self.XWavg = self.Xavg 
                self.XWstdev = self.Xstdev
                self.YWstdev = self.Ystdev
                #self.validW = False
        del self.S
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
        
def cluster(img, seed, minval, imgW, imgH, cluster_color ):
        #expect img[y][x] so max indicies are img[imgH-1][imgW-1]
        #accepted pixels must be light colored, with value >= minval
    c = Cluster(imgW, imgH)
    v = img[seed]
    if not is_cluster(v, minval): #if seed isn't a candidate 
        c.finalize()
        return c
    todo_list = [ (seed, v) ] # a fifo
    color_C = max(0,min(cluster_color, minval-1))  #new color of clustered pixels, must be less than minval (~162)
    img[seed] = color_C  
    while len(todo_list) > 0:
        center = todo_list[0]
        c.register(center)
        new_items = look_at_neighbors_swiss(img, center[0], minval, imgW, imgH)
        todo_list.extend(new_items )
        for item in new_items:
            img[item[0]] = color_C 
        del todo_list[0]
        if c.N >= 4096: #if we're over-clustering, break
            c.finalize()
            c.N = -c.N
            return c
    c.finalize()
    return c

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
            if is_cluster(img[y][x], minval):
                return True, (y,x)
       #Look South    
       y = y1-1
       for x in range(x0, x1, step):
            if is_cluster(img[y][x], minval):
                return True, (y,x)
       #Look East
       x = x0
       for y in range(y0, y1, step):
            if is_cluster(img[y][x], minval):
                return True, (y,x)
       #Look West
       x = x1-1
       for y in range(y0, y1, step):
            if is_cluster(img[y][x], minval):
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
