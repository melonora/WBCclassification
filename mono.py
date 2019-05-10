from transforms import *
from funcs import *
import random
import os
import math

ri = random.randint

def mono(showsteps = False):
    imsize = 120
    mask = np.zeros((imsize,imsize))
    minCell = 0.75
    maxCell = 0.90

    diam = ri(int(minCell*imsize), int(maxCell*imsize))

    p = (imsize/2-diam/2,imsize/2-diam/2)

    c = genCircle(diam)

    try: os.mkdir("../stapmono")
    except: pass
    mask[p[0]:p[0] + c.shape[0], p[1]:p[1]+c.shape[1]] += c*1./3
    mask = elastic(mask, 190, ri(8,12))
    if showsteps:misc.imsave('../stapmono/0.png', misc.imresize(mask,400))

    if random.random() <1./3:
        mask2 = np.zeros((imsize,imsize))
        diam2 = diam-ri(25,35)
        c2 = genCircle(diam2)
        p2 = (imsize/2-diam2/2+ri(0,10),imsize/2-diam2/2+ri(0,10))
        mask2[p2[0]:p2[0] + c2.shape[0], p2[1]:p2[1]+c2.shape[1]] += c2*2./3
        mask2 = elastic(mask2, 450, ri(12,15))
        mask += (mask>0)*(mask2>0)*mask2*2/3.
        if showsteps:misc.imsave('../stapmono/1.png', misc.imresize(mask,400))
    elif random.random()<1./3:
        mask2 = np.zeros((imsize+100,imsize+100))
        diam2 = diam/2 - 10
        diam3 = diam2+ri(-10,10)
        c2 = genCircle(diam2)
        c3 = genCircle(diam3)
        c4 = genCircle(diam2)
        angle = ri(0,360)
        angle2 = angle + ri(70,110)
        angle3 = angle2 + ri(70,110)

        p2 = (int(imsize/2+diam2/1.5*math.sin(math.radians(angle))-diam2/2)+50 , int(imsize/2+diam2/1.5*math.cos(math.radians(angle))-diam2/2)+50)
        p3 = (int(imsize/2+diam3/2*math.sin(math.radians(angle2))-diam3/2)+50 , int(imsize/2+diam3/2*math.cos(math.radians(angle2))-diam3/2)+50)
        p4 = (int(imsize/2+diam2/1.5*math.sin(math.radians(angle3))-diam2/2)+50 , int(imsize/2+diam2/1.5*math.cos(math.radians(angle3))-diam2/2)+50)
        mask2[p2[0]:p2[0] + c2.shape[0], p2[1]:p2[1]+c2.shape[1]] += c2
        mask2[p3[0]:p3[0] + c3.shape[0], p3[1]:p3[1]+c3.shape[1]] += c3
        mask2[p4[0]:p4[0] + c4.shape[0], p4[1]:p4[1]+c4.shape[1]] += c4
        mask2 = elastic(mask2, 190, ri(9,10))
        mask2[mask2>0] = 1
        mask2 = ((mask2[50:170,50:170] >.1)*(mask >.1)*(1-getEdge(mask,ri(5,8))[:,:,0])).astype(int)
        mask += mask2*2/3.
        if showsteps:misc.imsave('../stapmono/1.png', misc.imresize(mask,400))
    else:
        mask2 = np.zeros((imsize+100,imsize+100))
        diam2 = diam/2
        diam3 = diam2+ri(-10,10)
        c2 = genCircle(diam2)
        c3 = genCircle(diam3)
        angle = ri(0,360)
        angle2 = angle + ri(45,180)

        p2 = (int(imsize/2+diam2/2.5*math.sin(math.radians(angle))-diam2/2)+50 , int(imsize/2+diam2/2.5*math.cos(math.radians(angle))-diam2/2)+50)
        p3 = (int(imsize/2+diam3/2.5*math.sin(math.radians(angle2))-diam3/2)+50 , int(imsize/2+diam3/2.5*math.cos(math.radians(angle2))-diam3/2)+50)
        mask2[p2[0]:p2[0] + c2.shape[0], p2[1]:p2[1]+c2.shape[1]] += c2
        mask2[p3[0]:p3[0] + c3.shape[0], p3[1]:p3[1]+c3.shape[1]] += c3
        mask2 = elastic(mask2, 190, ri(9,10))
        mask2[mask2>0] = 1
        mask2 = ((mask2[50:170,50:170] >.1)*(mask >.1)*(1-getEdge(mask,ri(3,8))[:,:,0])).astype(int)
        mask += mask2*2/3.
        if showsteps:misc.imsave('../stapmono/1.png', misc.imresize(mask,400))

    cell = np.zeros((imsize,imsize,3))
    kleur(cell, mask<0.1, (1,1,1))
    kleur(cell, (mask>0.1)*(mask<.5), (243,187,220))
    kleur(cell, (mask>0.5), (214,144,233))
    if showsteps:cell[0,0] = 0
    if showsteps:misc.imsave('../stapmono/2.png', misc.imresize(cell,400))

    noise1 = minmax(genNoise(imsize,0.05, ri(0,1000))**2, 0.5, 1)
    noise2 = genNoise(imsize, 0.1)*genNoise(imsize, 0.1)
    noise2 = noise2 * 1/noise2.max()
    noise2 = minmax(1 - (getEdge(mask>.1, 8) * noise2),0.,1)
    noise3 = minmax(1-genSparseNoise(imsize, .4, .7, 1), 0.2, 1)
    noise4 = minmax(1-genSparseNoise(imsize, .2, .8, 1), 0.1, 1)

    cell = interpolation(cell, cfield(imsize, (145,116,191)), noise2, (mask>.1)*(mask<.5))
    if showsteps:misc.imsave('../stapmono/3.png', misc.imresize(cell,400))
    cell = interpolation(cell, cfield(imsize, (92,61,121)), noise3, (mask>.1)*(mask<.5))
    if showsteps:misc.imsave('../stapmono/4.png', misc.imresize(cell,400))
    cell = interpolation(cell, cfield(imsize, (1,1,1)), noise1, (mask>.1)*(mask<.5))
    if showsteps:misc.imsave('../stapmono/5.png', misc.imresize(cell,400))
    if random.random()>.5: cell = interpolation(cell, cfield(imsize, (255,250,255)), noise4, (mask>.1)*(mask<.5))
    if showsteps:misc.imsave('../stapmono/6.png', misc.imresize(cell,400))

    coreNoise1 = 1 - (getEdge(mask>.5, 9) * genNoise(imsize, 0.08,ri(0,1000)))
    radNoise2 = elastic(radialNoise(imsize),190,ri(2,3))[:,:,np.newaxis]
    coreNoise2 = minmax(1-genSparseNoise(imsize, .2, .8, 1), 0.1, 1)

    cell = interpolation(cell, cfield(imsize, (101,81,158)), coreNoise1, mask>.5)
    if showsteps:misc.imsave('../stapmono/7.png', misc.imresize(cell,400))
    cell = interpolation(cell, cfield(imsize, (145,70,190)), radNoise2, mask>.5)
    if showsteps:misc.imsave('../stapmono/8.png', misc.imresize(cell,400))
    if random.random() < .5: cell = interpolation(cell, cfield(imsize, (234,179,255)), coreNoise2, (mask>.5)*(radNoise2[:,:,0]>.2))
    if showsteps:misc.imsave('../stapmono/9.png', misc.imresize(cell,400))

    cell = gausBlur(cell, 1.5)
    cell[0,0] = 0
    if showsteps: misc.imsave('../stapmono/final.png', misc.imresize(cell,400))
    if not showsteps:
        misc.imsave('../dataset/mono/mask/%i.png'%i,mask)
        misc.imsave('../dataset/mono/cell/%i.png'%i,cell)

if __name__ == "__main__":
    mono(True)
    for i in xrange(100000):
        mono()
        if i%500 == 0:
    		print i
