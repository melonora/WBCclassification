from transforms import *
from funcs import *
import random
import os
import math

ri = random.randint

def lympho(showsteps = False):
    imsize = 120
    mask = np.zeros((imsize,imsize))

    try: os.mkdir("../staplympho")
    except: pass
    if random.random() < .5:
        minCell = 0.5
        maxCell = 0.55
        diam = ri(int(minCell*imsize), int(maxCell*imsize))
        diam2 = diam-ri(5,10)
        p = (imsize/2-diam/2,imsize/2-diam/2)
        p2 = (imsize/2-diam2/2,imsize/2-diam2/2)
        c = genCircle(diam)
        c2 = genCircle(diam2)

        mask[p[0]:p[0] + c.shape[0], p[1]:p[1]+c.shape[1]] += c*1./3
        mask[p2[0]:p2[0] + c2.shape[0], p2[1]:p2[1]+c2.shape[1]] += c2*2./3
        if showsteps:misc.imsave('../staplympho/0.png', misc.imresize(mask,400))
        mask = elastic(mask, 190, ri(12,18))
        if showsteps:misc.imsave('../staplympho/1.png', misc.imresize(mask,400))

        cell = np.ones((imsize,imsize,3))
        kleur(cell, (mask>0.5), (192,134,222))
        kleur(cell, (mask<.5)*(mask>.1),(132,102,232))

        coreNoise1 = 1 - (getEdge(mask>.5, 9) * genNoise(imsize, 0.08,ri(0,1000)))
        radNoise2 = elastic(radialNoise(imsize),190,ri(2,3))[:,:,np.newaxis]
        coreNoise2 = minmax(1-genSparseNoise(imsize, .2, .8, 1), 0.1, 1)

        if showsteps: cell[0,0] = (0,0,0)
        if showsteps:misc.imsave('../staplympho/3.png', misc.imresize(cell,400))
        cell = interpolation(cell, cfield(imsize, (101,81,158)), coreNoise1, mask>.5)
        if showsteps:misc.imsave('../staplympho/4.png', misc.imresize(cell,400))
        cell = interpolation(cell, cfield(imsize, (145,89,189)), radNoise2, mask>.5)
        if showsteps:misc.imsave('../staplympho/5.png', misc.imresize(cell,400))
        cell = interpolation(cell, cfield(imsize, (204,136,231)), coreNoise2, (mask>.5)*(radNoise2[:,:,0]>.2))
        if showsteps:misc.imsave('../staplympho/6.png', misc.imresize(cell,400))

        cell = gausBlur(cell, 1.5)
        cell[0,0] = (0,0,0)
        if showsteps:misc.imsave('../staplympho/final.png', misc.imresize(cell,400))

    else:
        minCell = 0.7
        maxCell = 0.8
        diam = ri(int(minCell*imsize), int(maxCell*imsize))
        diam2 = diam-ri(3,6)
        c = genCircle(diam)
        c2 = genCircle(diam2)
        p = (imsize/2-diam/2,imsize/2-diam/2)

        mask[p[0]:p[0] + c.shape[0], p[1]:p[1]+c.shape[1]] += c
        if showsteps:misc.imsave('../staplympho/0.png', misc.imresize(mask,400))
        mask = elastic(mask, ri(100,190), ri(9,12))
        angle = ri(0,360)
        p2 = (int(imsize/2+diam2/4*math.sin(math.radians(angle))-diam2/2)+40 , int(imsize/2+diam2/4*math.cos(math.radians(angle))-diam2/2)+40)
        mask2 = np.zeros((imsize+80,imsize+80))
        mask2[p2[0]:p2[0] + c2.shape[0], p2[1]:p2[1]+c2.shape[1]] += c2
        mask2 = elastic(mask2, 200, ri(8,12))

        mask2 = ((mask2[40:160,40:160] >.1)*(mask >.1)*(1-getEdge(mask,ri(1,8))[:,:,0])).astype(int)
        mask = mask*1/3.+mask2*2/3.
        if showsteps:misc.imsave('../staplympho/1.png', misc.imresize(mask,400))

        cell = np.ones((imsize,imsize,3))
        kleur(cell, (mask>0.5), (214,144,233))

        radNoise = elastic(radialNoise(imsize),190,ri(4,5))[:,:,np.newaxis]
        cytoNoiseEdge = 1 - (getEdge(mask>.1, 5) * genNoise(imsize, 0.08,ri(0,1000)))

        if showsteps: cell[0,0] = (0,0,0)
        if showsteps: misc.imsave('../staplympho/2.png', misc.imresize(mask,400))
        cell = interpolation(cell,cfield(imsize,(174,166,217)),cytoNoiseEdge,(mask<.5)*(mask>.1))
        if random.random() > .5: cell = interpolation(cell,cfield(imsize,(140,133,191)),1-radNoise,(mask<.5)*(mask>.1))
        else: cell = interpolation(cell,cfield(imsize,(198,194,236)),1-radNoise,(mask<.5)*(mask>.1))
        if showsteps: misc.imsave('../staplympho/3.png', misc.imresize(mask,400))

        coreNoise1 = 1 - (getEdge(mask>.5, 9) * genNoise(imsize, 0.08,ri(0,1000)))
        radNoise2 = elastic(radialNoise(imsize),190,ri(2,3))[:,:,np.newaxis]
        coreNoise2 = minmax(1-genSparseNoise(imsize, .2, .8, 1), 0.1, 1)

        cell = interpolation(cell, cfield(imsize, (101,81,158)), coreNoise1, mask>.5)
        if showsteps:misc.imsave('../staplympho/4.png', misc.imresize(cell,400))
        cell = interpolation(cell, cfield(imsize, (145,70,190)), radNoise2, mask>.5)
        if showsteps:misc.imsave('../staplympho/5.png', misc.imresize(cell,400))
        if random.random() < .5: cell = interpolation(cell, cfield(imsize, (234,179,255)), coreNoise2, (mask>.5)*(radNoise2[:,:,0]>.2))
        if showsteps:misc.imsave('../staplympho/6.png', misc.imresize(cell,400))

        cell = gausBlur(cell, 1.5)
        cell[0,0] = (0,0,0)
        if showsteps:misc.imsave('../staplympho/final.png', misc.imresize(cell,400))

    if not showsteps:
        misc.imsave('../dataset/lympho/mask/%i.png'%i,mask)
        misc.imsave('../dataset/lympho/cell/%i.png'%i,cell)




if __name__ == "__main__":
    for i in xrange(100000):
    	lympho()
        if i%500 == 0:
    		print i
