from multiprocessing import Pool
import time
def test(p, a):
    p = p+1
    time.sleep(10)
    print(p)
    return p
    
def WithMP():
    pool = Pool(processes=2)
    starttime = time.time()
    '''
    for i in range(2):
        pool.apply_async(func=test, args=(i,))   
    pool.close()
    pool.join()
    '''
    [c,d] = pool.starmap(test, [(0, 1),(1,1)])
    print("with MP time", time.time()-starttime)
    print("returned",c,d)

def NoMP():
    starttime = time.time()
    for i in range(2):
        test(i, 1)
    print("without MP time", time.time()-starttime)