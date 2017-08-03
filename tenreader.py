from itertools import zip_longest
# For Python 2, use izip_longest instead
class Tenfoldreader:
    'Output ten even subsets of original input'
    '''
    Read from the middle of the input,if the length of input is larger than or equal to 360
    The size of each subset will be 36, otherwise, size will be length // 10.
    Different methods give different output format, either list of list or list.
    '''
    
    def returnlist(self,strarr):
        # @param string list
        # @return string list
        # the length of return will be either 360 or the length of original list.
        hf = len(strarr) // 2 # Python 2 user no need to change this part, this is for Python 3
        if (hf >= 180):
            res = [None] * 360
            for i,string_ in enumerate (strarr[hf-180:hf+180]):
                res[i] = string_
            return res
        else:
            return strarr
    
    def returntupleoflist(self,strarr):
        # @param string list
        # @return tuple of string list
        # the length of return of always be 10.
        # but for the item length, it maybe less than 36.
        hf = len(strarr) // 2
        res = []
        if (hf >= 180):
            tmp = []
            for i, string_ in enumerate (strarr[hf-180:hf+180]):
                tmp.append(string_)
                if ((i +1)% 36 == 0):
                    res.append(tmp)
                    tmp = []
            return res
        else:
            args = [iter(strarr)] * 10
            return list(zip_longest(*args))
        
    def chunks(self,iterableitem, n):
        '''Yield successive n-sized chunks from iterableitem.'''
        for i in range(0, len(iterableitem), n):
            yield iterableitem[i:i + n]
            


    
a = ['1','2','3','4','5','6','7','8','9','10','11','12']
print(type(a))
tf = Tenfoldreader()
b = tf.returntupleoflist(a)
