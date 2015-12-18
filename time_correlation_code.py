import numpy as np
import sys
import time
import skxray.core.roi as roi
from matplotlib import gridspec

import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
mcolors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k','darkgoldenrod','oldlace', 'brown','dodgerblue'   ])
markers = itertools.cycle(list(plt.Line2D.filled_markers))
lstyles = itertools.cycle(['-', '--', '-.','.',':'])

#Dec 1, NSLS-II, yugangzhang, yuzhang@bnl.gov

def autocor_one_time( num_buf,  ring_mask, imgs, num_lev=None, start_img=None, end_img=None, bad_images = None, threshold=None):   
    start_time = time.time()
    #print (dly)
    if start_img is None:
        start_img=0
    if end_img is None:
        try:
            end_img= len(imgs)
        except:
            end_img= imgs.length
            
    #print (start_img, end_img)    
    noframes = end_img - start_img #+ 1
    #print (noframes)
    
    if num_lev is None:num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    nolev = num_lev
    nobuf =num_buf
    print ( 'The lev number is %s'%num_lev)
    
    dly, dict_dly = delays( num_lev, num_buf, time=1 )
    #print (dly.max())
    lev_leng = np.array( [  len(  dict_dly[i] ) for i in list(dict_dly.keys())   ])
    
    qind, pixelist = roi.extract_label_indices(   ring_mask  )
    noqs = np.max(qind)    
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]
    nopixels = nopr.sum()     
    start_time = time.time() 
   
    buf =  np.ma.zeros([num_lev,num_buf,nopixels])
    buf.mask = True   
            
    
    cts=np.zeros(num_lev)
    cur=np.ones(num_lev) * num_buf
    countl = np.array( np.zeros(  num_lev ),dtype='int')  
    
    g2 =  np.zeros( [ noframes, noframes, noqs] )   
    
    G=np.zeros( [(nolev+1)*int(nobuf/2),noqs])
    IAP=np.zeros( [(nolev+1)*int(nobuf/2),noqs])
    IAF=np.zeros( [(nolev+1)*int(nobuf/2),noqs])
    num= np.array( np.zeros(  num_lev ),dtype='int')  
    
    Num= { key: [0]* len(  dict_dly[key] ) for key in list(dict_dly.keys())  }
    print ('Doing g2 caculation of %s frames---'%(noframes ))
    ttx=0     
    #if bad_images is None:bad_images=[]
    for n in range( start_img, end_img ):   ##do the work here
        
        img = imgs[n] 
        img_ = (np.ravel(img))[pixelist]
        
        #print ( img_.max() )
        if threshold is not None:
            if img_.max() >= threshold:
                print ('bad image: %s here!'%n)
                img_ =  np.ma.zeros( len(img_) )
                img_.mask = True    
                
        if bad_images is not None:        
            if n in bad_images:
                print ('bad image: %s here!'%n)
                img_ =  np.ma.zeros( len(img_) )
                img_.mask = True 
        
        
        cur[0]=1+cur[0]%num_buf  # increment buffer  
 
        buf[0, cur[0]-1 ]=  img_

        img=[] #//save space 
        img_=[]
        countl[0] = 1+ countl[0]
 
        process_one_time(lev=0, bufno=cur[0]-1,
            G=G,IAP=IAP,IAF=IAF, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly, Num=Num, lev_leng=lev_leng )     
        #time_ind[0].append(  current_img_time   )
        processing=1
        lev=1
        while processing:
            if cts[lev]:
                prev=  1+ (cur[lev-1]-1-1+num_buf)%num_buf
                cur[lev]=  1+ cur[lev]%num_buf
                countl[lev] = 1+ countl[lev] 
 
                bufa = buf[lev-1,prev-1]
                bufb=  buf[lev-1,cur[lev-1]-1] 
                
                if (bufa.data==0).all():
                    buf[lev,cur[lev]-1] =  bufa
                elif (bufb.data==0).all():
                    buf[lev,cur[lev]-1] = bufb 
                else:
                    buf[lev,cur[lev]-1] = ( bufa + bufb ) /2. 
                
                cts[lev]=0                
                t1_idx=   (countl[lev]-1) *2
 
                process_one_time(lev=lev, bufno=cur[lev]-1,
                        G=G,IAP=IAP,IAF=IAF, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly,Num =Num, lev_leng=lev_leng )     
 
                lev+=1
                #//Since this level finished, test if there is a next level for processing
                if lev<num_lev:processing = 1
                else:processing = 0                                
            else:
                cts[lev]=1      #// set flag to process next time
                processing=0    #// can stop until more images are accumulated              
 
        
        if  n %(noframes/10) ==0:
            sys.stdout.write("#")
            sys.stdout.flush()                
    #print G.shape    
    if (len(np.where(IAP==0)[0])!=0) and ( 0 not in nopr):
        gmax = np.where(IAP==0)[0][0]        
    else:
        gmax=IAP.shape[0]
    #g2=G/(IAP*IAF)
    #print G
    g2=(G[:gmax]/(IAP[:gmax]*IAF[:gmax]))       
    elapsed_time = time.time() - start_time
    #print (Num)
    print ('Total time: %.2f min' %(elapsed_time/60.))        
    return  g2,dly[:gmax]  #, elapsed_time/60.


    
            
def process_one_time(lev, bufno,    
                     G,IAP,IAF, buf, num, num_buf,noqs,qind,nopr, dly,Num,lev_leng ):
    
    num[lev]+=1     
    if lev==0:imin=0
    else:imin= int(num_buf/2 )         
    for i in range(imin, min(num[lev],num_buf) ):        
        ptr=lev*int(num_buf/2)+i    
        delayno=int( (bufno-i)%num_buf) #//cyclic buffers 
        
        IP  =  buf[lev,delayno]
        IF  =  buf[lev,bufno]  
        ind = ptr - lev_leng[:lev].sum()        
        IP_ = IP.copy()
        IF_ = IF.copy()
        
        if (IP.data ==0).all():            
            IF_=np.zeros( IP.shape )
            IP_= np.zeros( IP.shape )
            Num[lev+1][ind] += 1                    
        if (IF.data ==0).all():  
            #print ('here IF =0')
            IF_ =  np.zeros( IF.shape )
            IP_=   np.zeros( IF.shape ) 
            if (IP.data ==0).all():
                pass                
            else:
                Num[lev+1][ind] += 1          
        norm_num = num[lev]-i -  Num[lev+1][ind]  
        
        #print (   lev, ptr, num[lev]-i, Num[lev+1][ind] )
        #print (ind, lev_leng)
        
        if not (IP_ ==0).all():
            G[ptr]+=  (   np.bincount(qind,      weights= IF_*IP_ )[1:]/nopr- G[ptr] )/ norm_num
            IAP[ptr]+= (  np.bincount(qind,   weights= IP_)[1:]/nopr-IAP[ptr] )/ norm_num
            IAF[ptr]+= (  np.bincount(qind,   weights= IF_)[1:]/nopr-IAF[ptr] )/ norm_num
 

def autocor_two_time( num_buf,  ring_mask, imgs, num_lev=None, start_img=None, end_img=None    ):
    

    #print (dly)
    if start_img is None:start_img=0
    if end_img is None:
        try:
            end_img= len(imgs)
        except:
            end_img= imgs.length
            
    #print (start_img, end_img)    
    noframes = end_img - start_img #+ 1
    #print (noframes)
    
    if num_lev is None:num_lev = int(np.log( noframes/(num_buf-1))/np.log(2) +1) +1
    print ( 'The lev number is %s'%num_lev)
    
    dly, dict_dly = delays( num_lev, num_buf, time=1 )
    #print (dly.max())
    
    qind, pixelist = roi.extract_label_indices(   ring_mask  )
    noqs = np.max(qind)    
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]
    nopixels = nopr.sum() 
    
    start_time = time.time()
    
    buf=np.zeros([num_lev,num_buf,nopixels])  #// matrix of buffers, for store img
    
    
    cts=np.zeros(num_lev)
    cur=np.ones(num_lev) * num_buf
    countl = np.array( np.zeros(  num_lev ),dtype='int')  
    
    g12 =  np.zeros( [ noframes, noframes, noqs] )      
    
    num= np.array( np.zeros(  num_lev ),dtype='int')          
    time_ind ={key: [] for key in range(num_lev)}   
    
    ttx=0        
    for n in range( start_img, end_img ):   ##do the work here
        
        cur[0]=1+cur[0]%num_buf  # increment buffer  
        img = imgs[n] 
        
        #print ( 'The insert image is %s' %(n) )
    
        buf[0, cur[0]-1 ]=  (np.ravel(img))[pixelist]
        img=[] #//save space 
        countl[0] = 1+ countl[0]
        current_img_time = n - start_img +1
    
        process_two_time(lev=0, bufno=cur[0]-1,n=current_img_time,
                        g12=g12, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly)     
        time_ind[0].append(  current_img_time   )
        processing=1
        lev=1
        while processing:
            if cts[lev]:
                prev=  1+ (cur[lev-1]-1-1+num_buf)%num_buf
                cur[lev]=  1+ cur[lev]%num_buf
                countl[lev] = 1+ countl[lev]                                
                buf[lev,cur[lev]-1] = ( buf[lev-1,prev-1] + buf[lev-1,cur[lev-1]-1] ) /2.
                cts[lev]=0                
                t1_idx=   (countl[lev]-1) *2
                current_img_time = ((time_ind[lev-1])[t1_idx ] +  (time_ind[lev-1])[t1_idx +1 ] )/2. 
                time_ind[lev].append(  current_img_time      )  
                process_two_time(lev=lev, bufno=cur[lev]-1,n=current_img_time,
                        g12=g12, buf=buf, num=num, num_buf=num_buf, noqs=noqs, qind=qind, nopr=nopr, dly=dly)  
                lev+=1
                #//Since this level finished, test if there is a next level for processing
                if lev<num_lev:processing = 1
                else:processing = 0                                
            else:
                cts[lev]=1      #// set flag to process next time
                processing=0    #// can stop until more images are accumulated              
 
        
        if  n %(noframes/10) ==0:
            sys.stdout.write("#")
            sys.stdout.flush()                
    
    
    for q in range(noqs):            
        x0 =  g12[:,:,q]
        g12[:,:,q] = np.tril(x0) +  np.tril(x0).T - np.diag( np.diag(x0) )            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    
    return g12, elapsed_time/60.



    
    
    
            
def process_two_time(lev, bufno,n ,    
                     g12, buf, num, num_buf,noqs,qind,nopr, dly ):
    num[lev]+=1  
    if lev==0:imin=0
    else:imin= int(num_buf/2 )
    for i in range(imin, min(num[lev],num_buf) ):
        ptr=lev*int(num_buf/2)+i    
        delayno=(bufno-i)%num_buf #//cyclic buffers            
        IP=buf[lev,delayno]
        IF=buf[lev,bufno]
        I_t12 =  (np.histogram(qind, bins=noqs, weights= IF*IP))[0]
        I_t1  =  (np.histogram(qind, bins=noqs, weights= IP))[0]
        I_t2  =  (np.histogram(qind, bins=noqs, weights= IF))[0]
        tind1 = (n-1)
        tind2=(n -dly[ptr] -1)
        
        if not isinstance( n, int ):                
            nshift = 2**(lev-1)                
            for i in range( -nshift+1, nshift +1 ):
                #print tind1+i
                g12[ int(tind1 + i), int(tind2 + i) ] =I_t12/( I_t1 * I_t2) * nopr
        else:
                #print tind1
            g12[ tind1, tind2 ]  =   I_t12/( I_t1 * I_t2) * nopr       
        
        
        

    
def delays( num_lev=3, num_buf=4, time=1 ): 
    ''' DOCUMENT delays(time=)
        return array of delays.
        KEYWORD:  time: scale delays by time ( should be time between frames)
     '''
    if num_buf%2!=0:print ("nobuf must be even!!!"    )
    dly=np.zeros( (num_lev+1)*int(num_buf/2) +1  )        
    dict_dly ={}
    for i in range( 1,num_lev+1):
        if i==1:imin= 1
        else:imin= int(num_buf/2)+1
        ptr=(i-1)*int(num_buf/2)+ np.arange(imin,num_buf+1)
        dly[ptr]= np.arange( imin, num_buf+1) *2**(i-1)            
        dict_dly[i] = dly[ptr-1]            
        dly*=time
        #print (i, ptr, imin)
    return dly, dict_dly
            