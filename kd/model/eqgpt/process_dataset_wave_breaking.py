import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
def get_data(cell_refs):
    all_data=[]
    frame_rate = 20.0
    for i in range(1,len(cell_refs)):
        t=i/frame_rate
        ref = cell_refs[i]
        # 通过引用读取具体的数据
        frame_data = f[ref]
        # 将数据转为 numpy 数组
        frame_array = np.array(frame_data).T
        t_array=np.ones_like(frame_array[:,0])*t
        t_array=t_array.reshape(-1,1)
        frame_array=np.hstack((t_array,frame_array))
        all_data.append(frame_array)

        # plt.plot(frame_array[:,1],frame_array[:,2])
        # plt.show()
    all_data=np.concatenate(all_data,axis=0)
    return all_data



if __name__ == '__main__':
    filename = 'Surface_Chen.mat'
    with h5py.File(filename, 'r') as f:
    surface_struct=f['Surface']
    case_name=list(surface_struct.keys())
    print("case name:", case_name)
    data_dict={}
    for name in case_name:
        l_field = surface_struct[name]
        surf_global = l_field[b'Surface_global']
        cell_refs = np.array(surf_global).squeeze()
        all_data=get_data(cell_refs)
        data_dict[name]=all_data

        surf_global_interp = l_field[b'Surface_global_interp']

    pickle.dump(data_dict,
                open(f'wave_breaking_data.pkl', 'wb'))
