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
            
     
    

class Get_Pixel_Array(object):
    def __init__(self, indexable, pixelist):
        self.indexable = indexable
        self.pixelist = pixelist
        self.shape = indexable.shape
        try:
            self.length= len(indexable)
        except:
            self.length= indexable.length           
            
    def get_data(self ): 
        #print (self.length)
        data_array = np.zeros([ self.length,len(self.pixelist)])
        for key in range(self.length ):
            data_array[key] = np.ravel( self.indexable[key])[self.pixelist]  
        return data_array
    
    

    
 
    
    
    
def autocor_arrays_two_time( seg1, pixelist,qind, seg2=None,                         
                            get_half=False,get_whole=False,up_half=True,
            print_=True,    ):
        #for same seg1 and seg2, use get_half=True, get_whole=True

        if seg2 is None:
            seg2=seg1 #half_flag=True #half_flag for only calculate half

        start_time = time.time()    
        m,n = seg1.shape
        #print m,n
        noqs = len( np.unique(qind) )
        nopr = np.bincount(qind, minlength=(noqs+1))[1:] #qind start from 1   
        seg1f = np.ravel( seg1 )                
        qinds = np.zeros( [ m, len(pixelist)],dtype=int)
        for i in range( m ):qinds[i]=qind + ( max(qind) ) *i     #qind start from 1   
        G_t1 = np.bincount( qinds.ravel(), weights = seg1f )[1:]      #qind start from 1          
        G_t1= G_t1.reshape( [m, noqs] )      
        
        g12s =  np.zeros( [m,m, noqs] )
        cal=-1
        
        for i in range( m ):
            if get_half:
                termin=i+1;
            else:
                termin=m
                #print ('not get half')
            if up_half:
                #print ('cal up_half')
                seg12i = ( seg1[:termin] * seg2[i]).ravel()
                qindsi=qinds[:termin]
            else: #for down_half
                seg12i = ( seg1[termin-1 : ] * seg2[i]).ravel()
                qindsi=qinds[: m- (termin-1) ]
            G_t12i = np.bincount( qindsi.ravel(), weights = seg12i )[1:]
            #print G_t12i.shape, qindsi.flatten().shape, seg12i.shape
            #print seg12i.shape, G_t12i.shape            
            mG = (G_t12i.shape)[0]
            #print seg1[ termin-1  : ].shape            
            G_t12i= G_t12i.reshape( [ int(mG/noqs), noqs] )            
            seg2i =  np.bincount( qind.ravel(), weights = seg2[i] )[1:]

            #print seg12i.shape,qindsi.shape, mG,G_t12i.shape,seg2i.shape
            
            #print G_t12i.shape            
            #print G_t12i.shape, seg2i.shape, G_t1.shape, termin, g12s.shape           
            #print termin-1
            if up_half:
                g12s[i][:termin] = G_t12i/( G_t1[:termin] * seg2i ) * nopr
            else:
                g12s[i][termin-1:] = G_t12i/( G_t1[termin-1:] * seg2i ) * nopr
            
            if print_:
                if  int(i /(m/10.)) >cal:                
                    sys.stdout.write("#")
                    sys.stdout.flush()
                    cal = int(i /(m/10.))
                    #elapsed_time = time.time() - start_time
                    #print ('%s:  cal time: %.2f min' %( int(i /(m/10.)),elapsed_time/60.)  )
                    
##        if up_half==False:
##            g12s_=g12s.copy()
##            g12s=g12s*0
##            for q in range(noqs):            
##                x0 =  g12s_[:,:,q]                
##                g12s[:,:,q] =   tril(x0).T 
                
            
        if get_whole:
            for q in range(noqs):  
                #print (q)
                x0 =  g12s[:,:,q]
                if up_half:
                    g12s[:,:,q] = np.tril(x0) +  np.tril(x0).T - np.diag(np.diag(x0))
                else:
                    g12s[:,:,q] = np.triu(x0) +  np.triu(x0).T - np.diag(np.diag(x0))
        if print_:
            elapsed_time = time.time() - start_time
            print ('Total time: %.2f min' %(elapsed_time/60.)  )
            
        return g12s



class Reverse_Coordinate(object):
    def __init__(self, indexable, mask):
        self.indexable = indexable
        self.mask = mask
        self.shape = indexable.shape
        self.length= len(indexable)
    def __getitem__(self, key ):      
        if self.mask is not None:
            img =self.indexable[key] * self.mask  
        else:
            img = self.indexable[key]
            
        if len(img.shape) ==3:
            img_=img[:,::-1,:]
        if len(img.shape)==2:
            img_=img[::-1,:] 
        return img_
 

def autocor_large_arrays_two_time( data, pixelist, qind, divide=4,  
                 only_one_nodia=True,  get_whole=True,print_=True  ): 
    #currently, data is the pims data
    
    #noframes/divde should be int
    try:
        noframes = len(data)
    except:
        noframes = data.length    
    if noframes/divide - int( noframes/divide ) != 0:
        print ('noframes/divde should be int!!! Please give another divide number!!!')          
    start_time = time.time() 
    noqs = len( np.unique(qind) )
    g12L =  np.zeros( [noframes,noframes, noqs] )
    step_fram = int(noframes/divide)    
    data_div = np.zeros( [step_fram, len(pixelist), divide])    
    for block in range( divide):
        imgs_ = data[block*step_fram : (block+1)* step_fram]
        #print (imgs_.shape)
        imgsr = Reverse_Coordinate(imgs_, mask=None)
        data_div[:,:,block]= Get_Pixel_Array( imgsr, pixelist).get_data()
    
    #print (data_div.shape)
        
    cal=-1
    m=0
    st = divide**2/20.
    
    for block1 in range(divide):
        data1 = data_div[:,:,block1]
        for block2 in range(block1, divide):
            #print (block1, block2)
            if print_:
                m+=1
                if  int( m/st ) >cal:                
                    sys.stdout.write("#")
                    sys.stdout.flush()
                    cal = int( m/st  ) 
                        
            if block1==block2: #this is the diagonal part
                #print (block1, block2)
                fm1,fm2 = [block1*step_fram , (block1+1)* step_fram]
                #print fm1,fm2
                g12L[fm1:fm2,fm1:fm2,:]= autocor_arrays_two_time(                        
                        seg1 = data1, pixelist=pixelist,qind=qind, seg2=None, 
                        get_half=True,get_whole=False,up_half=True,
                        print_= False)  

            else:  #this is the no-diagonal part              
                if not only_one_nodia:  #cal all nodiagon
                    data2 = data_div[:,:,block2]
                    fm1,fm2 = [block1*step_fram , (block1+1)* step_fram]
                    fm3,fm4 = [block2*step_fram , (block2+1)* step_fram]
                    g12L[fm3:fm4,fm1:fm2,:] =autocor_arrays_two_time(    
                        seg1=data1, pixelist=pixelist,qind=qind,seg2=data2,
                        get_half=False, get_whole=False,  up_half=True,                      
                       print_= False)
                else: # cal only one nodiagon
                    block2x=block1+1
                    data2 = data_div[:,:,block2x]
                    fm1,fm2 = [block1*step_fram , (block1+1)* step_fram]
                    fm3,fm4 = [block2x*step_fram , (block2x+1)* step_fram]
                    g12L[fm3:fm4,fm1:fm2,:] =autocor_arrays_two_time(
                        seg1=data1,pixelist=pixelist,qind=qind,seg2=data2,
                        get_half=True,get_whole=False,up_half=False,
                       print_= False)
                    
    if get_whole:
        for q in range(noqs):  
            #print (q)
            x0 =  g12L[:,:,q]
            g12L[:,:,q] = np.tril(x0) +  np.tril(x0).T - np.diag(np.diag(x0))                   
    if print_:
        elapsed_time = time.time() - start_time
        print ('Total time: %.2f min' %(elapsed_time/60.)  )

    return g12L








def auto_two_Array1( data, box_mask,   ):
    start_time = time.time()
    qind, pixelist = roi.extract_label_indices(   box_mask  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]
    
    #print (nopr)
    
    #data_pixel =   Get_Pixel_Array( data, pixelist).get_data()
    try:
        noframes = len(data)
    except:
        noframes = data.length
    g12b = np.zeros(  [noframes, noframes, noqs] )
    Unitq = (noqs/10)
    proi=0
    for qi in range(1, noqs + 1 ):
        pixelist_qi = pixelist[ np.where( qind == qi)[0] ]
        data_pixel_qi =   Get_Pixel_Array( data, pixelist_qi).get_data()
        sum1 = (np.average( data_pixel_qi, axis=1)).reshape( 1, noframes   )  
        sum2 = sum1.T
        g12b[:,:,qi -1 ] = np.dot(   data_pixel_qi, data_pixel_qi.T)  /sum1  / sum2  / nopr[qi -1]
        #print ( proi, int( qi //( Unitq) ) )
        if  int( qi //( Unitq) ) == proi:
            sys.stdout.write("#")
            sys.stdout.flush() 
            proi += 1
            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    return g12b



def test():
    pixelist_q1 = pixelist[ np.where( qind ==1)[0] ]
    seg_q1 =   Get_Pixel_Array( imgsr, pixelist_q1).get_data()
    nopr_q1 = len(pixelist_q1)
    sum1 = (np.average( seg_q1, axis=1)).reshape( 1, seg_q1.shape[0]   )  
    sum2 = sum1.T
    m= np.dot(   seg_q1, seg_q1.T)  /sum1  / sum2  / nopr_q1


def get_mean_intensity( data_pixel, qind):
    noqs = len( np.unique(qind) )
    mean_inten = {}
               
    for qi in range(1, noqs + 1 ):
        pixelist_qi =  np.where( qind == qi)[0] 
        #print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi =    data_pixel[:,pixelist_qi]  
        mean_inten[qi] =  data_pixel_qi.sum( axis =1 )
    return  mean_inten
    
    
    
def auto_two_Array_g1_norm( data, box_mask, data_pixel=None  ):
    start_time = time.time()
    qind, pixelist = roi.extract_label_indices(   box_mask  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]    
    if data_pixel is None:
        data_pixel =   Get_Pixel_Array( data, pixelist).get_data()
        #print (data_pixel.shape)
    
    try:
        noframes = len(data)
    except:
        noframes = data.length
    g12b_norm = np.zeros(  [noframes, noframes, noqs] )
    g12b = np.zeros(  [noframes, noframes, noqs] )
    norms = np.zeros(  [noframes, noqs] )
    
    Unitq = (noqs/10)
    proi=0
    
    for qi in range(1, noqs + 1 ):
        pixelist_qi =  np.where( qind == qi)[0] 
        #print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi =    data_pixel[:,pixelist_qi]   
        
        sum1 = (np.average( data_pixel_qi, axis=1)).reshape( 1, noframes   )  
        sum2 = sum1.T        
        #norms_g12  =  sum1 * sum2 * nopr[qi -1]
        norms[:,qi -1 ]  =  sum1 
        
        g12b[:,:,qi -1 ] = np.dot(   data_pixel_qi, data_pixel_qi.T)  
        g12b_norm[:,:,qi -1 ] = g12b[:,:,qi -1 ]/ sum1 / sum2 / nopr[qi -1]
        #print ( proi, int( qi //( Unitq) ) )
        if  int( qi //( Unitq) ) == proi:
            sys.stdout.write("#")
            sys.stdout.flush() 
            proi += 1
            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    return g12b_norm, g12b, norms



def auto_two_Array( data, box_mask, data_pixel=None  ):
    start_time = time.time()
    qind, pixelist = roi.extract_label_indices(   box_mask  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]    
     
    if data_pixel is None:
        data_pixel =   Get_Pixel_Array( data, pixelist).get_data()
        #print (data_pixel.shape)
    
    try:
        noframes = len(data)
    except:
        noframes = data.length
    g12b = np.zeros(  [noframes, noframes, noqs] )
    Unitq = (noqs/10)
    proi=0
    
    for qi in range(1, noqs + 1 ):
        pixelist_qi =  np.where( qind == qi)[0] 
        #print (pixelist_qi.shape,  data_pixel[qi].shape)
        data_pixel_qi =    data_pixel[:,pixelist_qi]   
        
        sum1 = (np.average( data_pixel_qi, axis=1)).reshape( 1, noframes   )  
        sum2 = sum1.T       
        
        g12b[:,:,qi -1 ] = np.dot(   data_pixel_qi, data_pixel_qi.T)  /sum1  / sum2  / nopr[qi -1]
        #print ( proi, int( qi //( Unitq) ) )
        if  int( qi //( Unitq) ) == proi:
            sys.stdout.write("#")
            sys.stdout.flush() 
            proi += 1
            
    elapsed_time = time.time() - start_time
    print ('Total time: %.2f min' %(elapsed_time/60.))
    
    return g12b


def get_one_time_from_two_time(  g12, norms=None, nopr = None, timeperframe=1.0  ):
    m,n,noqs = g12.shape
    g2f12 = []       
    for q in  range(noqs):   
        temp=[]    
        y=g12[:,:,q]        
        for tau in range(m): 
            if norms is None:
                temp.append( np.diag(y,k=int(tau)).mean() )
            else:
                yn = norms[:,q]
                 
                yn1 =  np.average( yn[tau:] )
                yn2 =  np.average( yn[: m-tau] )                  
                temp.append(  np.diag(y,k=int(tau)).mean()/  (yn1*yn2*nopr[q])   )
        temp = np.array( temp).reshape( len(temp),1)
        if q==0:
            g2f12 =  temp
        else:
            g2f12=np.hstack( [g2f12,  temp] ) 
    return g2f12


def get_g2_from_g12_0( g12, slice_num = 6, slice_width=5, slice_start=None, slice_end=None  ):
    '''g12 is the two-time correlation data
        slice_num is the slice number of the diagonal of g12
        slice_width is the slice width
        
        M,N = g12.shape,
        slice can start from 1 to 2*N-1
    '''
    m,n = g12.shape
    age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
                     qwidth = slice_width, noqs =slice_num  )    
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)      
    g2 = {}
    for i,age in enumerate(age_center):        
        age_edges_0, age_edges_1 = age_edge[ i*2 : 2*i+2]        
        age_edges_0 -= n
        age_edges_1 -= n            
        min_edge = max(  [abs(age_edges_0), abs(age_edges_1 )] )
        min_len =  len( np.diag( g12[::-1,:], min_edge  ) )
        for j, bin_age in enumerate( range( age_edges_0, age_edges_1 )):            
            diag = np.diag( g12[::-1,:], bin_age  )[:min_len].reshape(1,min_len)
            if j ==0: 
                g2_age =   diag  
                 
            else:                 
                g2_age = np.vstack( [g2_age,  diag] )
                
        g2[age] = (g2_age.mean(axis = 0)[:int(min_len/2)+1])[::-1]
    return g2 


def get_qedge( qstart,qend,qwidth,noqs,  ):
    ''' DOCUMENT make_qlist( )
    give qstart,qend,qwidth,noqs
    return a qedge by giving the noqs, qstart,qend,qwidth.
           a qcenter, which is center of each qedge 
    KEYWORD:  None    ''' 
    import numpy as np 
    qcenter = np.linspace(qstart,qend,noqs)
    #print ('the qcenter is:  %s'%qcenter )
    qedge=np.zeros(2*noqs) 
    qedge[::2]= (  qcenter- (qwidth/2)  ) #+1  #render  even value
    qedge[1::2]= ( qcenter+ qwidth/2) #render odd value
    return qedge, qcenter  

def get_g2_from_g12_1( g12, slice_num = 6, slice_width=5, slice_start=None, slice_end=None  ):
    '''g12 is the two-time correlation data
        slice_num is the slice number of the diagonal of g12
        slice_width is the slice width
        
        M,N = g12.shape,
        slice can start from 1 to 2*N-1
    '''
    m,n = g12.shape
    age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
                     qwidth = slice_width, noqs =slice_num  )    
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)      
    g2 = {}
    for i,age in enumerate(age_center):        
        age_edges_0, age_edges_1 = age_edge[ i*2 : 2*i+2]        
        age_edges_0 -= n
        age_edges_1 -= n   
        
        max_edge = min(  [abs(age_edges_0), abs(age_edges_1 )] )
        max_len =  len( np.diag( g12[::-1,:], max_edge  ) )
        Diag = []
        leng = age_edges_1 -  age_edges_0
        arr = np.ma.empty((max_edge, leng ))
        arr.mask = True        
        for j, bin_age in enumerate( range( age_edges_0, age_edges_1 )):            
            diag = np.diag( g12[::-1,:], bin_age  )
            arr[ :len(diag), j  ] = diag                
        g2[age] = (arr.mean(axis = 0)[:int(max_len/2)+1])[::-1]
        
        
        age_edges_0 -= n
        age_edges_1 -= n   
        #print (age_edges_0, age_edges_1)
        if age_edges_0*age_edges_1 <=0:
            max_edge=0
        else:
            max_edge = min(  [abs(age_edges_0), abs(age_edges_1 )] )
         
        max_len =  len( np.diag( g12[::-1,:], max_edge  ) )
        #print (max_edge,max_len)
        Diag = []
        leng = age_edges_1 -  age_edges_0
        arr = np.ma.empty(( leng, max_len ))
        #print (arr.shape)
        arr.mask = True        
        for j, bin_age in enumerate( range( age_edges_0, age_edges_1 )):            
            diag = np.diag( g12[::-1,:], bin_age  )
            #print (arr)
            #print (diag.shape, diag.ravel())
            #print  (diag.ravel())
            print (len(diag))
            if len(diag)%2==0:
                len_j = int( len(diag/2) -1 )
            else:
                len_j = int( len(diag)/2 + 1 )
            apd = diag.ravel()[:len_j][::-1]  
            print (apd)
            arr[ j, :len(apd) ] = apd
        #print (arr)
        g2[age] = np.array((arr.mean(axis = 0)))
    return g2 
 



def trans_g12_to_square( g12 ):
    '''g12 is two-time-correlation data, for one Q
       M,N = g12.shape
       transform g12 into a masked array with shape as (N, 2N-1)
    
    '''
    M,N = g12.shape
    arr = np.ma.empty(( 2*N-1,N ))
    arr.mask = True
    for i in range(N):
        arr[i:(2*N-1-i):2, i  ] = g12.diagonal(i)
    return arr


######
def make_g12_mask(g12b, badlines):
    '''make g12 maks to mask teh badlines'''
    
    m,n,qs = g12b.shape
    #g12b_mask = np.ma.empty( ( m,n ) )
    g12b_mask = np.ma.ones( ( m,n ) )
    g12b_mask.mask= False    
    for bdl in badlines:
        g12b_mask.mask[:,bdl] = True
        g12b_mask.mask[bdl,:] = True
    return g12b_mask
        
def masked_g12( g12b, badlines):
    '''mask g12 by badlines'''
    
    m,n,qs = g12b.shape
    g12b_ = np.ma.empty_like( g12b )
    g12b_mask = make_g12_mask(g12b, badlines)
    for i in range(qs):
        g12b_[:,:,i] = g12b[:,:,i] * g12b_mask
    return g12b_




def get_g2_from_g12( g12, slice_num = 6, slice_width=5, slice_start=None, slice_end=None  ):
    '''g12 is the two-time correlation data
        slice_num is the slice number of the diagonal of g12
        slice_width is the slice width
        
        M,N = g12.shape,
        slice can start from 1 to 2*N-1
    '''
    arr= trans_g12_to_square( g12 )
    m,n = arr.shape #m should be 2*n-1
    age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
                     qwidth = slice_width, noqs =slice_num  )    
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)  
    #print (age_edge, age_center)
    g2 = {}
    for i,age in enumerate(age_center):         
        age_edges_0, age_edges_1 = age_edge[ i*2 : 2*i+2]  
        g2i = arr[ age_edges_0: age_edges_1   ].mean( axis =0 )
        g2i_ = np.array( g2i )       
        g2[age] =   g2i_[np.nonzero( g2i_)[0]]
        
    return g2 

def show_g12_age_cuts( g12, g12_num=16, slice_num = 6, slice_width= 500,  slice_start=500,
                      slice_end=5000-1,timeperframe=1,vmin= 1, vmax= 1.25 ):
    data = g12[:,:,g12_num]
    g2 = get_g2_from_g12( data, slice_num, slice_width,slice_start, slice_end )
    age_edge, age_center = get_qedge( qstart=slice_start, qend= slice_end, 
                                     qwidth = slice_width, noqs = slice_num )
    print ('the cut age centers are: ' +str(age_center)     )
    M,N = data.shape

    #fig, ax = plt.subplots( figsize = (8,8) )
    
    figw =10
    figh = 10
    fig = plt.figure(figsize=(figw,figh)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 8],height_ratios=[8,8]   ) 
    ax = plt.subplot(gs[0])     
    ax1 = plt.subplot(gs[1])     
    
    im=ax.imshow( data, origin='lower' , cmap='viridis', 
             norm= LogNorm( vmin, vmax ) , extent=[0, N, 0, N ] )

    linS = []
    linE=[]
    linS.append( zip( [0]*len(age_center), np.int_(age_center) ))
    linE.append( zip(  np.int_(age_center), [0]*len(age_center) ))
    for i, [ps,pe] in enumerate(zip(linS[0],linE[0])):     
        if ps[1]>=N:s0=ps[1] - N;s1=N
        else:s0=0;s1=ps[1]        
        if pe[0]>=N:e0=N;e1=pe[0] - N
        else:e0=pe[0];e1=0     
        lined= slice_width/2.  #in data width
        linewidth=    (lined * (figh*72./N)) * 0.8
        ax.plot( [s0,e0],[s1,e1], linewidth=linewidth ,alpha=0.3 )  #, color=   )  
    
    ax.set_title(  '%s_frames'%(N)    )
    ax.set_xlabel( r'$t_1$ $(s)$', fontsize = 18)
    ax.set_ylabel( r'$t_2$ $(s)$', fontsize = 18)
    fig.colorbar(im)
     
    
    
    ax1.set_title("Aged_G2")
    for i in sorted(g2.keys()):
        gx= np.arange(len(g2[i])) * timeperframe
        marker = next(markers)        
        ax1.plot( gx,g2[i], '-%s'%marker, label=r"$age= %.1f s$"%(i*timeperframe))
        ax1.set_ylim( vmin, vmax )
        ax1.set_xlabel(r"$\tau $ $(s)$", fontsize=18) 
        ax1.set_ylabel("g2")
        ax1.set_xscale('log')
    ax1.legend(fontsize='small', loc='best' ) 
    plt.show()
    
    
    return g2



def get_tau_from_g12( g12, slice_num = 6, slice_width=1, slice_start=None, slice_end=None  ):
    '''g12 is the two-time correlation data
        slice_num is the slice number of the diagonal of g12
        slice_width is the slice width
        
        M,N = g12.shape,
        slice can start from 1 to 2*N-1
    '''
    arr= trans_g12_to_square( g12 )
    m,n = arr.shape #m should be 2*n-1
    age_edge, age_center = get_qedge( qstart=slice_start,qend= slice_end,
                     qwidth = slice_width, noqs =slice_num  )    
    age_edge, age_center = np.int_(age_edge), np.int_(age_center)  
    #print (age_edge, age_center)
    tau = {}
    for i,age in enumerate(age_center):         
        age_edges_0, age_edges_1 = age_edge[ i*2 : 2*i+2] 
        #print (age_edges_0, age_edges_1)
        g2i = arr[ :,age_edges_0: age_edges_1   ].mean( axis =1 )
        g2i_ = np.array( g2i )       
        tau[age] =   g2i_[np.nonzero( g2i_)[0]]
        
    return tau

def show_g12_tau_cuts( g12, g12_num=16, slice_num = 6, slice_width= 1,  slice_start=1,
                      slice_end=2000-1,timeperframe=1, draw_scale_tau =10,vmin= 1, vmax= 1.25 ):
    data = g12[:,:,g12_num]
    tau = get_tau_from_g12( data, slice_num, slice_width,slice_start, slice_end )
    age_edge, age_center = get_qedge( qstart=slice_start, qend= slice_end, 
                                     qwidth = slice_width, noqs = slice_num )
    print ('the cut tau centers are: ' +str(age_center)     )
    M,N = data.shape

    #fig, ax = plt.subplots( figsize = (8,8) )
    
    figw =10
    figh = 10
    fig = plt.figure(figsize=(figw,figh)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 8],height_ratios=[8,8]   ) 
    ax = plt.subplot(gs[0])     
    ax1 = plt.subplot(gs[1])     
    
    im=ax.imshow( data, origin='lower' , cmap='viridis', 
             norm= LogNorm( vmin= vmin, vmax= vmax ) , extent=[0, N, 0, N ] )

    linS = []
    linE=[]
    linS.append( zip(  np.int_(age_center) -1, [0]*len(age_center)   ))
    linE.append( zip(  [N -1]*len(age_center), N  - np.int_(age_center)   ))
    for i, [ps,pe] in enumerate(zip(linS[0],linE[0])):
        lined= slice_width/2. *draw_scale_tau  #in data width
        linewidth=    (lined * (figh*72./N)) * 0.8
        #print (ps,pe)
        ax.plot( [ps[0],pe[0]],[ps[1],pe[1]], linewidth=linewidth ) #, color=   )  
    
    ax.set_title(  '%s_frames'%(N)    )
    ax.set_xlabel( r'$t_1$ $(s)$', fontsize = 18)
    ax.set_ylabel( r'$t_2$ $(s)$', fontsize = 18)
    fig.colorbar(im)    
    
    ax1.set_title("Tau_Cuts_in_G12")
    for i in sorted(tau.keys()):
        gx= np.arange(len(tau[i])) * timeperframe
        marker = next(markers)        
        ax1.plot( gx,tau[i], '-%s'%marker, label=r"$tau= %.1f s$"%(i*timeperframe))
        ax1.set_ylim( vmin,vmax )
        ax1.set_xlabel(r'$t (s)$',fontsize=5)
        ax1.set_ylabel("g2")
        ax1.set_xscale('log')
    ax1.legend(fontsize='small', loc='best' ) 
    plt.show()
    
    
    return tau


def his_tau(tau, hisbin=20, plot=True,timeperframe=1):
    his={}
    for key in list(tau.keys()):
        his[key] = np.histogram( tau[key], bins=hisbin)
        
    if plot:            
        fig, ax1 = plt.subplots(figsize=(8, 8))        
        ax1.set_title("Tau_histgram")
        for key in sorted(his.keys()):
            tx= 0.5*( his[key][1][:-1] + his[key][1][1:])
            marker = next(markers)       
            ax1.plot( tx, his[key][0], '-%s'%marker, label=r"$tau= %.1f s$"%(key*timeperframe) )
            #ax1.set_ylim( 1.05,1.35 )
            ax1.set_xlim( 1.05,1.35 )
            ax1.set_xlabel(r'$g_2$',fontsize=19)
            ax1.set_ylabel(r"histgram of g2 @ tau",fontsize=15)
            #ax1.set_xscale('log')
        ax1.legend(fontsize='large', loc='best' ) 
        plt.show()
    
        
    return his
        















