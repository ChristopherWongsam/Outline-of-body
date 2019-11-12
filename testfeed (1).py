#import paramiko
import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def getcood(img):
    edges=img
    coodx=[]
    coody=[]
    width,height=edges.shape
    for i in range(height):
        for j in range(width):
            if  np.any(edges[j,i] == [255,255,255]):
                coody.append(i)
                coodx.append(j)
    return coody,coodx
def converttoedge(img):
        edges = cv2.Canny(img,100,200)
        return edges
def sendcoodx(var):
    outFx = open("myOutFilex.txt", "w")
    for line in var:
        # write line to output file
        outFx.write(str(line)+",")
    outFx.close()
    
def sendcoody(var):
    outFx = open("myOutFiley.txt", "w")
    for line in var:
        # write line to output file
        outFx.write(str(line)+",")
    outFx.close()
def foregroundextract(img,xpos,ypos,wid,height):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (xpos,ypos,wid,height)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img
 
def convertdst(img):
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(frame,-1,kernel)   
    return dst
    

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
#    ssh = paramiko.SSHClient()
#    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#    ssh.connect('192.168.137.99', username="pi", password="raspberry")
#    sftp = ssh.open_sftp()
#    localpath = 'myOutFilex.txt'
#    remotepath = '/home/crestelsetup/patchzip/myOutFilex.txt'
#    sftp.put(localpath, remotepath)
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    edges1 = converttoedge(frame)

    dst = convertdst(frame)
    edgesdst=converttoedge(dst)
    fgmask = fgbg.apply(dst)
    fgmaskdst = fgbg.apply(edgesdst)
    edges2 = converttoedge(dst)
    edges3 = converttoedge(fgmask)
#   frame->dst->edges->backgroundrem
    foreext=foregroundextract(frame,150,50,400,425)
    foreedges=converttoedge(foreext)
#   Getting and sending x and y coordinate
    y,x=getcood(foreedges)
    sendcoodx(x)    
    sendcoody(y)
#    outFx = open("myOutFilex.txt", "w")
#    outFy = open("myOutFiley.txt", "w")
#    for line in x:
        # write line to output file
#        outFx.write(str(line)+",")
#    outFx.close()
#    for line in y:
        # write line to output file
#        outFy.write(str(line)+",")
    localpath = "myOutFilex.txt"
#    remotepath = '/home/pi/Adafruit_Python_MCP4725/examples/myOutFilex.txt'
#    sftp.put(localpath, remotepath)
#    outFy.close()
#    outFx.close()
    cv2.imshow('Original',foreext)
    cv2.imshow('edgesdst',edgesdst)
    cv2.imshow('dst',dst)
    cv2.imshow('edges2',foreedges)
    plt.plot(x, y,'ro',markersize=1)
    plt.xlabel('X')
    plt.ylabel('Y')

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()