import os
import sys
import numpy as np
# print(cv2.__version__)
from time import time
# import glob
import cv2

from scipy.linalg import svd
from scipy.linalg import cholesky



def vanishing_pair(src,dst):
    #helps in finding two perpendicular direction's vanishing pt in image_plane
    h,_ = cv2.findHomography(src, dst, cv2.RANSAC,5.0)
    d1 =np.array([1,0,0],dtype =np.float32).reshape(-1,1)
    d2 =np.array([0,1,0],dtype =np.float32).reshape(-1,1)
    v1=h @ d1
    v2 =h @ d2
    return list(v1.reshape(-1)),list(v2.reshape(-1))

def get_whole_vanishing_pts(path,files):
    #collect  vanishing pt pair from each image
    vpts=[]

    checkerboard =(6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    c=0 #count
    src = objp[:,:,:-1].reshape(-1,1,2) #obj2d pts 

    for i in files:
        img = cv2.imread(os.path.join(path,i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cond ,corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if cond == True:
            dst = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)#corners
            
            # vanishing_pair
            v1,v2 =vanishing_pair(src,dst)
            v1=v1/v1[-1]
            v2=v2/v2[-1]
            vpts.append([v1,v2])
            c+=1
        else:
            print("cond wrong in img")
    print('used img count ',c)

    return vpts

def svd_helper(mtx):
    # Compute the SVD of the matrix
    m,n=mtx.shape
    U, S, VT = svd(mtx)
    params = VT[-1,:]
    return params

def compute_intrinsic_params(vpts): #vpts - total_vanishing pts


    #method and related eqn involved is -
    #v2.T @ W @v1
    # here v1,v2 is vanishing pt pair and W =K_inv.T @K_inv
    #then svd(DLT) and further cholesky decomposition for finding K

    mtx_w =[] #W matrix 
    npts =len(vpts)
    for i in range(npts):
        l=[0 for j in range(6)]
        u1 ,v1  =vpts[i][0][0],vpts[i][0][1]
        u2 ,v2  =vpts[i][1][0],vpts[i][1][1]

        #fill l acc to eqn of pt
        l[0]=u1*u2
        l[1]=(u1*v2) + (v1*u2)
        l[2]=u1+u2
        l[3]=v1*v2
        l[4]=v1+v2
        l[5]=1

        mtx_w.append(l)

    # mtx1 SVD - U D V.T
    ws = svd_helper(np.array(mtx_w,dtype =np.float32)).reshape(-1)

    #construct matrix mw that is Kinv.T @Kinv
    mw =[[ws[0],ws[1],ws[2]],
    [ws[1],ws[3],ws[4]],
    [ws[2],ws[4],ws[5]]]

    kkt = -1*np.linalg.inv(np.array(mw,dtype =np.float32))

    #cholesky
    k_t= np.linalg.cholesky(kkt)
    k =k_t.T
    return k

def get_intrinsic_params(path,files):

    if (calibrate==0):
        k = [[8.0254254e+02, 1.6457086e+02, 6.5770942e-01],
        [0.0000000e+00, 7.0912805e+02, 4.1176298e-01],
        [0.0000000e+00, 0.0000000e+00, 1.0000001e+00]]

        k =np.array(k,dtype =np.float32)
    else:
        vpts = get_whole_vanishing_pts(path,files)
        k = compute_intrinsic_params(vpts)
    return k

def get_extrinsic_params(k,p):
    #P is homography matrix of image
    h1=p[:,0]
    h2 =p[:,1]
    h3 =p[:,-1]
    #R =[r1,r2,r3]
    #k [r1,r2,t] = p 
    #below computation to find r1,r2 and t -
    k_inv= np.linalg.inv(k)
    r1=k_inv @ h1.reshape(-1,1)
    factor = 1/np.linalg.norm(r1)
    r1=factor *r1
    r2=k_inv @ h2.reshape(-1,1)
    r2=factor * r2
    r3=np.cross(r1.reshape(-1),r2.reshape(-1))
    r3=r3.reshape(-1,1)
    t =k_inv @ h3.reshape(-1,1)
    t=factor*t
    R =np.hstack((r1,r2,r3))
    # print(R)
    new_p = k @ np.hstack((R,t))
    #P =K [R |t]
    return new_p

def get_projection(P,X):
    #find projection of 3d world homogenous coordinate to 2d coordinates of image
    x =P @ np.array(X).reshape(-1,1)
    x=x.reshape(-1)
    x=x/x[-1]
    return [x[0],x[1]]

def helper_cube(img,imgpts):
    #helpful in rendering of artificial cube 
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw top floor in yellow
    img = cv2.drawContours(img, [imgpts[4:8]],-1,(0,255,255),-3)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def draw_cube_surface(img,P,K):
    #place artificial cube on chessboard surface
    #cube with side 4 with one coord at (1,1,0,1)
    cube_corners= [[1, 1, 0, 1], [3, 1, 0, 1], [3, 3, 0, 1], [1, 3, 0, 1], [1, 1, 2, 1], [3, 1, 2, 1], [3, 3, 2, 1], [1, 3, 2, 1]]

    imgpts =[]
    n =len(cube_corners)
    for i in range(n):
        imgpts.append(get_projection(P,cube_corners[i]))
    
    imgpts = np.array(imgpts)
    imgpts = (np.rint(imgpts)).astype(int)

    image =helper_cube(img,imgpts)
    return image


def place_object_on_images(path,files):
    #helpful in placing object in given images
    checkerboard =(6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    src = objp[:,:,:].reshape(-1,1,3) #obj2d
    ncorners =checkerboard[0] * checkerboard[1]

    k =get_intrinsic_params(path,files)
    for i in files:
        img = cv2.imread(os.path.join(path,i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cond ,corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if cond == True:
            dst = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)#corners

            p,_ =cv2.findHomography(src,dst)

            P=get_extrinsic_params(k,p)

            image = draw_cube_surface(img,P,k)
            cv2.imwrite(i+"_cube.jpeg",image)

        else:
            print("cond wrong in img")




if __name__ == "__main__":
    #argv1- path of input folder containing images

    path = sys.argv[1]
    files =os.listdir(path)
    tf =len(files) 

    calibrate =0 #do not find K ,intrinsic params
    place_object_on_images(path,files)
