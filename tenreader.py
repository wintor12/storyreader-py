from itertools import zip_longest
# For Python 2, use izip_longest instead
class Tenfoldreader:
    'Output ten even subsets of original input'
    '''
    Read from the middle of the input,if the length of input is larger than or equal to 360
    The size of each subset will be 36, otherwise, size will be length // 10.
    Different methods give different output format, either list of list or list.
    '''
    
    def returnlist(self,strlist):
        # @param string list
        # @return string list
        # the length of return will be either 360 or the length of original list.
        hf = len(strlist) // 2 # Python 2 user no need to change this part, this is for Python 3
        if (hf >= 180):
            res = [None] * 360
            for i,string_ in enumerate (strlist[hf-180:hf+180]):
                res[i] = string_
            return res
        else:
            res = self.returnlistoflist(strlist)
            ret = []
            for ele in res:
                for item in ele:
                    ret.append(item)
            return ret
    
    def returnlistoflist(self,strlist):
        # @param string list
        # @return tuple of string list
        # the length of return of always be 10.
        # but for the item length, it maybe less than 36.
        hf = len(strlist) // 2
        res = []
        if (hf >= 180):
            tmp = []
            for i, string_ in enumerate (strlist[hf-180:hf+180]):
                tmp.append(string_)
                if ((i +1)% 36 == 0):
                    res.append(tmp)
                    tmp = []
            return res
        else:
            res = list(self.spliteven(strlist,10))
            return self.resizeTo36each(res)
        
#     def chunks(self,iterableitem, n):
#         '''Yield successive n-sized chunks from iterableitem.'''
#         for i in range(0, len(iterableitem), n):
#             yield iterableitem[i:i + n]
    
#     def grouper(n, iterable, fillvalue=None):
#     "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
#         args = [iter(iterable)] * n
#         return izip_longest(fillvalue=fillvalue, *args)
    
    def spliteven(self,inputlist, n):
        k, m = divmod(len(inputlist), n)
        return (inputlist[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    def resizeTo36each(self,inputlist):
        '''resize every element of the list to 36'''
        # @param the length of inputlist is 10
        # @return each element will have size 36, thus the 'total' will be 360
        # add tail'blank' to element of which size is less than 36
        # add both head && tail 'blank' is silly, not good
        for element in inputlist:
            if (len(element)<36):
                while (len(element)<36):
                    element.append(' ')
        return inputlist

    
a = ['1','2','3','4','5','6','7','8','9','10','11','12']
c = list(range(36))
print(type(a))
tf = Tenfoldreader()
# b = tf.returntupleoflist(c)
# print(len(b))
d = tf.returnlistoflist(c)
