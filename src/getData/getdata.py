
import numpy as np

import h5py
import glob

fs_phi_abs2 = glob.glob("../phi_abs2.*.h5")


i00 = 0
all_data_phiabs2 = np.array([])
z_array = np.array([],dtype=np.int32)
for f_name in fs_phi_abs2:
    f_phiabs2 = h5py.File(f_name,"r")
    

    data_phiabs2 = f_phiabs2.get("FloatArray")[:]

    num_t = 1020
   
    if(i00==0):
        all_data_phiabs2 = np.append(all_data_phiabs2, data_phiabs2[num_t,:])

    else:
        all_data_phiabs2 = np.vstack([all_data_phiabs2,data_phiabs2[num_t,:]])

    i00 += 1

    print f_name
    z_array = np.append(z_array,np.int32(f_name.split('.')[-2]))

    print "I am at: ",i00


# re-sorting
z_array_sort = z_array.argsort()

z_array = z_array[z_array_sort]
all_data_phiabs2 = all_data_phiabs2[z_array_sort,:]

np.save("all_data_phiabs2",all_data_phiabs2)
np.save("z_array",z_array)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.figure(0)
plt.imshow(all_data_phiabs2,cmap=cm.jet,aspect='auto')
plt.colorbar()
plt.show()



