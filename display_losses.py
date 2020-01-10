

import matplotlib.pyplot as plt

import re

import sys


if(len(sys.argv)==1):
    filepath = 'log_store_case_2.txt'
    print('we are printing losses in default file name')
else:
    filepath = str(sys.argv[1])



with open(filepath) as fp:
   line = fp.readline()
   cnt = 0

   gen_losses=[]
   disc_losses=[]
   rad_losses=[]
   cyc_losses=[]
   iden_losses=[]
   while line:
       #print("Line {}: {}".format(cnt, line.strip()))

       string=line.strip()
       #print(string)
       if (string.startswith('Epoch')):
           cnt+=1
       if (string.startswith('Iter') and cnt % 3 == 0):
        gen_loss = re.search('Generator Loss:(.*)Discrimator Loss', string).group(1)
        disc_loss = re.search('Discrimator Loss:(.*)GA2B', string).group(1)
        #rad_loss = re.search('radial:(.*)G_cyc', string).group(1)
        #cycle_loss = re.search('G_cyc:(.*)G_id', string).group(1)
        #iden_loss = string.split("G_id:",1)[1]

        gen_losses.append(float ( gen_loss )) 
        disc_losses.append(float(disc_loss))
        #rad_losses.append(float(rad_loss))
        #cyc_losses.append(float(cycle_loss))
        #iden_losses.append(float(iden_loss))
        #print(gen_loss,' ',disc_loss,' ',rad_loss,' ',cycle_loss,' ',iden_loss)  

       line = fp.readline()

    
#print(gen_losses)


print(len(gen_losses))
plt.plot(gen_losses)
plt.plot(disc_losses)
#plt.plot(rad_losses)
#plt.plot(cyc_losses)
#plt.plot(iden_losses)

plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend(['gen_loss', 'disc_loss','rad','cyc','iden'], loc='upper left')
plt.show()
