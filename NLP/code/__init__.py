from analyze import Analyze
# import codecs
# text = codecs.open('D:\PyCharm Community Edition 2016.3.2\DeeCamp\\NLPpart\data\CarReport\CarReport_122.txt', 'r', 'gb18030').read()

ana = Analyze()
# ana = ana.init()
# print(1)
keywords = ana.keywords
pos = ana.pos
cws = ana.cws