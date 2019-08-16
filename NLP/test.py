import sys
sys.path.append('./code/')

import code
import codecs
text = codecs.open('../CarReport_6.txt', 'r', 'utf-8').read()


keyword = code.keywords(text,10)
print(keyword)
pos = code.pos(keyword)
print(pos)
