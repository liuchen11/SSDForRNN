require 'util.randSVD'
require 'util.vectorNorm'

randM=torch.randn(2000,1000)
begin=os.clock()
u1,s1,v1=torch.svd(randM)
finish=os.clock()
print('take',finish-begin,'seconds')
begin=os.clock()
u2,s2,v2=randSVD:svd(randM,200)
finish=os.clock()
print('take',finish-begin,'seconds')
--Schatten infinity tends to be much more accurate
acc_s1=vectorNorm:norm(s1,1)
acc_si=vectorNorm:norm(s1,1/0)
app_s1=vectorNorm:norm(s2,1)
app_si=vectorNorm:norm(s2,1/0)
print('accurate s1',acc_s1)
print('approximate s1',app_s1)
print('accurate si',acc_si)
print('approximate si',app_si)